"""Train V2 final + DeepStack adapters on DenseFusion, then optionally UCF."""
import argparse
import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset

import train_qwen3_showo2_adapter as base
from train_qwen3_showo2_adapter_stage2_ucf import UCFImageDataset
from qwen3_showo2_deepstack import DeepStackAdapter, unpack_visual, feature_loss, load_adapter

from tqdm import tqdm
# ============================================================
# CONFIG — edit these settings directly
# ============================================================

STAGE = 1



OUTPUT_DIR = "./showo2_qwen3_adapter_stage1_v2"

# None: start with pretrained Show-o/Qwen and random adapters.
# Set a checkpoint path to initialize from previous training.
INIT_CHECKPOINT = None

BATCH_SIZE = 12
EPOCHS = 10

# None: use the stage-specific learning rate.
LR = None

DEEPSTACK_WEIGHT = 1.0


@torch.no_grad()
def targets(images, visual, processor):
    images = [base.resize_and_pad_image(im, (base.QWEN_SIZE, base.QWEN_SIZE)) for im in images]
    inputs = processor.image_processor(images=images, return_tensors="pt", do_resize=False)
    final, deep = unpack_visual(visual(
        inputs["pixel_values"].to(device=base.DEVICE, dtype=base.DTYPE),
        grid_thw=inputs["image_grid_thw"].to(base.DEVICE),
    ))
    # Fixed square images give equal token counts; verify rather than silently truncate.
    if final.shape[0] % len(images):
        raise ValueError("Unequal Qwen token grids")
    final = final.reshape(len(images), -1, final.shape[-1])
    deep = [d.reshape_as(final) for d in deep]
    return final, deep


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", type=int, choices=[1, 2], default=STAGE)
    p.add_argument("--output", default=OUTPUT_DIR)
    p.add_argument("--init-checkpoint", default=INIT_CHECKPOINT,
                   help="V1 warm start for stage 1; trained V2 for stage 2")
    p.add_argument("--densefusion-json", default=base.DENSEFUSION_JSON)
    p.add_argument("--image-root", default=base.IMAGE_ROOT)
    p.add_argument("--train-root")
    p.add_argument("--val-root")
    p.add_argument("--vae-path", default=base.WAN_VAE_PATH)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--deepstack-weight", type=float, default=DEEPSTACK_WEIGHT)
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or not 0 < args.deepstack_weight < float("inf"):
        p.error("epochs, batch-size and deepstack-weight must be positive")
    random.seed(base.SEED)
    torch.manual_seed(base.SEED)
    base.WAN_VAE_PATH = args.vae_path
    if args.stage == 1:
        ds = base.DenseFusionDataset(args.densefusion_json, args.image_root)
        if len(ds) < 2:
            p.error("Need at least two images for train/validation")
        ids = list(range(len(ds)))
        random.Random(base.SEED).shuffle(ids)
        n = max(1, int(len(ds) * base.VAL_FRACTION))
        train, val = Subset(ds, ids[n:]), Subset(ds, ids[:n])
    else:
        if not all([args.train_root, args.val_root, args.init_checkpoint]):
            p.error("Stage 2 requires --train-root, --val-root and --init-checkpoint")
        train, val = UCFImageDataset(args.train_root), UCFImageDataset(args.val_root)
        if set(map(lambda x: str(Path(x).resolve()), train.paths)) & set(map(lambda x: str(Path(x).resolve()), val.paths)):
            p.error("Train and validation paths overlap")
    loaders = [DataLoader(ds, batch_size=args.batch_size, shuffle=(i == 0), collate_fn=base.collate_fn)
               for i, ds in enumerate([train, val])]
    checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=True) if args.init_checkpoint else None
    if args.stage == 2 and checkpoint.get("fusion_version") != 2:
        p.error("Stage 2 requires a V2 checkpoint; train DeepStack heads in stage 1 first")
    visual, processor = base.load_qwen_visual()
    showo, vae = base.load_showo2_visual()
    config = dict(showo_dim=showo.fusion_proj[-1].out_features,
                  qwen_dim=visual.config.out_hidden_size,
                  levels=len(visual.deepstack_visual_indexes))
    adapter = DeepStackAdapter(**config).to(device=base.DEVICE, dtype=base.DTYPE)
    if checkpoint:
        showo.fusion_proj.load_state_dict(checkpoint["fusion_proj"])
        if checkpoint.get("fusion_version") == 2:
            if checkpoint["adapter_config"] != config:
                raise ValueError("Checkpoint/model dimensions differ")
            if (checkpoint["qwen_model"] != base.QWEN3_MODEL or
                    checkpoint["deepstack_indexes"] != list(visual.deepstack_visual_indexes)):
                raise ValueError("Checkpoint Qwen model or DeepStack indexes differ")
            adapter = load_adapter(checkpoint, base.DEVICE, base.DTYPE)
        else:
            adapter.proj.load_state_dict({k.removeprefix("proj."): v for k, v in checkpoint["adapter"].items()})
            print("V1 warm start: final head restored; DeepStack heads start randomly")
    for param in showo.parameters():
        param.requires_grad_(False)
    if args.stage == 1:
        showo.fusion_proj.requires_grad_(True)
    lr = args.lr if args.lr is not None else (1e-4 if args.stage == 1 else 1e-5)
    groups = [{"params": adapter.parameters(), "lr": lr}]
    if args.stage == 1:
        groups.append({"params": showo.fusion_proj.parameters(), "lr": base.LR_FUSION})
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    trainable = [p for g in optimizer.param_groups for p in g["params"]]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.glob("checkpoint*.pt")):
        raise FileExistsError("Use a fresh output directory to preserve previous checkpoints")
    best = float("inf")
    for epoch in range(args.epochs):
        metrics = {}
        for training, loader in zip([True, False], loaders):
            adapter.train(training)
            showo.eval()
            showo.fusion_proj.train(training and args.stage == 1)
            sums = [0.0, 0.0, 0.0]
            count = 0

            phase = "Train" if training else "Validation"

            pbar = tqdm(
                loader,
                desc=f"Epoch {epoch + 1}/{args.epochs} | {phase}",
                dynamic_ncols=True,
            )

            for batch in pbar:
                images = []
                for item in batch:
                    with Image.open(item["path"]) as im:
                        images.append(im.convert("RGB"))
                target = targets(images, visual, processor)
                with torch.set_grad_enabled(training):
                    raw = base.get_showo2_features(images, showo, vae, deterministic_vae=True)
                    aligned = base.spatial_align_showo_to_qwen(raw, target[0])
                    losses = feature_loss(adapter(aligned), target, args.deepstack_weight)
                    if training:
                        optimizer.zero_grad(set_to_none=True)
                        losses[0].backward()
                        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                        optimizer.step()
                count += len(images)
                sums = [s + v.item() * len(images) for s, v in zip(sums, losses)]
                pbar.set_postfix(
                    loss=f"{losses[0].item():.4f}",
                    final=f"{losses[1].item():.4f}",
                    deep=f"{losses[2].item():.4f}",
                    avg=f"{sums[0] / count:.4f}",
                )
            metrics["train" if training else "val"] = [s / count for s in sums]
        print(json.dumps({"epoch": epoch + 1, **metrics}), flush=True)
        state = dict(fusion_version=2, stage=args.stage, epoch=epoch + 1,
                     adapter_config=config, adapter=adapter.state_dict(),
                     fusion_proj=showo.fusion_proj.state_dict(), optimizer=optimizer.state_dict(),
                     qwen_model=base.QWEN3_MODEL, deepstack_indexes=list(visual.deepstack_visual_indexes),
                     args=vars(args), metrics=metrics)
        torch.save(state, out / f"checkpoint_epoch_{epoch + 1}.pt")
        if metrics["val"][0] < best:
            best = metrics["val"][0]
            torch.save(state, out / "checkpoint_best.pt")


if __name__ == "__main__":
    main()