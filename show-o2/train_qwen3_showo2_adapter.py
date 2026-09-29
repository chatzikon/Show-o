import os
import gc
import json
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
)

from models import Showo2Qwen2_5, WanVAE
from datasets.utils import resize_and_pad_image, to_tensor_and_normalize


# ============================================================
# CONFIG
# ============================================================

DENSEFUSION_JSON = (
    "/home/chatziko/datasets/DenseFusion-4V-100K/"
    "DenseFusion-4V-100k/DenseFusion-4V-100k.jsonl"
)

IMAGE_ROOT = (
    "/home/chatziko/datasets/DenseFusion-4V-100K/"
    "images_extracted"
)

SHOWO2_MODEL = "showlab/show-o2-1.5B"

QWEN3_MODEL = "Qwen/Qwen3-VL-4B-Instruct"

WAN_VAE_PATH = "/home/chatziko/PycharmProjects/PythonProject/Show-o/show-o2/Wan2.1_VAE.pth"

OUTPUT_DIR = "./showo2_qwen3_adapter_stage1"

SHOWO_SIZE = 432
QWEN_SIZE = 448

BATCH_SIZE = 12
GRAD_ACCUM_STEPS = 1
EPOCHS = 10

LR_ADAPTER = 1e-4
LR_FUSION = 1e-5

VAL_FRACTION = 0.02
SEED = 42

DEVICE = "cuda"
DTYPE = torch.bfloat16


# ============================================================
# DATASET
# ============================================================

class DenseFusionDataset(Dataset):

    def __init__(self, json_file, image_root):
        self.image_root = Path(image_root)

        print("Indexing DenseFusion images...")

        image_index = {}

        for p in self.image_root.rglob("*"):
            if p.suffix.lower() in {
                ".jpg", ".jpeg", ".png", ".webp"
            }:
                image_index[p.stem] = str(p)

        print(f"Found {len(image_index)} image files")

        self.samples = []

        with open(json_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)

                image_id = str(item["image_id"])

                if image_id in image_index:
                    self.samples.append(
                        {
                            "image_id": image_id,
                            "path": image_index[image_id],
                        }
                    )

        print(
            f"Matched {len(self.samples)} "
            f"DenseFusion annotations to images"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def collate_fn(batch):
    return batch


# ============================================================
# ADAPTER
# ============================================================

class Showo2ToQwenAdapter(nn.Module):

    def __init__(self):
        super().__init__()

        self.proj = nn.Linear(
            1536,
            2560,
        )

    def forward(self, x):
        return self.proj(x)


# ============================================================
# LOAD QWEN3 VISUAL ENCODER ONLY
# ============================================================

def load_qwen_visual():

    print("Loading Qwen3-VL-4B...")

    processor = AutoProcessor.from_pretrained(
        QWEN3_MODEL
    )

    # Load complete checkpoint on CPU first.
    full_model = (
        Qwen3VLForConditionalGeneration
        .from_pretrained(
            QWEN3_MODEL,
            dtype=DTYPE,
            low_cpu_mem_usage=True,
            device_map="cpu",
        )
    )

    # Keep only the visual encoder.
    visual = full_model.model.visual

    full_model.model.visual = None

    del full_model

    gc.collect()

    visual = visual.to(
        device=DEVICE,
        dtype=DTYPE
    )

    visual.eval()

    for p in visual.parameters():
        p.requires_grad = False

    return visual, processor


# ============================================================
# LOAD SHOW-O2 VISUAL PATH
# ============================================================

def load_showo2_visual():

    print("Loading Show-o2...")

    model = Showo2Qwen2_5.from_pretrained(
        SHOWO2_MODEL,
        use_safetensors=False,
        load_llm=False,
        load_siglip_pretrained=False
    )

    # Freeze everything first.
    for p in model.parameters():
        p.requires_grad = False

    # Only Spatial Fusion will be trained.
    for p in model.fusion_proj.parameters():
        p.requires_grad = True

    # We don't need Show-o2's language model or generation head.
    for name in [
        "diffusion_head_a",
        "diffusion_head_b",
        "time_embed",
        "diff_proj",
        "time_embed_proj",
    ]:
        if hasattr(model, name):
            delattr(model, name)

    gc.collect()

    model = model.to(
        device=DEVICE,
        dtype=DTYPE
    )

    model.eval()

    # fusion_proj must stay in training mode.
    model.fusion_proj.train()

    print("Loading Wan VAE...")

    vae = WanVAE(
        vae_pth=WAN_VAE_PATH,
        dtype=DTYPE,
        device=DEVICE,
    )

    return model, vae


# ============================================================
# QWEN FEATURES
# ============================================================

@torch.no_grad()
def get_qwen_features(
    images,
    visual,
    processor,
):

    # Use a fixed square resolution.
    # 448 / 32 = 14 -> 14 x 14 = 196 final visual tokens.
    images = [
        resize_and_pad_image(
            img,
            (QWEN_SIZE, QWEN_SIZE)
        )
        for img in images
    ]

    inputs = processor.image_processor(
        images=images,
        return_tensors="pt",
        do_resize=False,
    )

    pixel_values = inputs["pixel_values"].to(
        DEVICE,
        dtype=DTYPE,
    )

    grid_thw = inputs["image_grid_thw"].to(DEVICE)

    outputs = visual(
        pixel_values,
        grid_thw=grid_thw,
        return_dict=True,
    )

    # After Qwen's visual merger:
    # total_visual_tokens x 2560
    z = outputs.pooler_output

    batch_size = len(images)

    z = z.reshape(
        batch_size,
        -1,
        2560,
    )

    return z


# ============================================================
# SHOW-O2 FEATURES
# ============================================================

def get_showo2_features(
    images,
    model,
    vae,
        deterministic_vae=False
):

    tensors = []

    for img in images:

        img = resize_and_pad_image(
            img,
            (SHOWO_SIZE, SHOWO_SIZE),
        )

        x = to_tensor_and_normalize(
            img,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        )

        tensors.append(x)

    x = torch.stack(tensors).to(
        DEVICE,
        dtype=DTYPE,
    )

    # Show-o2 VAE expects temporal dimension.
    x = x.unsqueeze(2)

    # Everything before fusion_proj stays frozen.
    with torch.no_grad():

        z = vae.sample(x, deterministic=deterministic_vae)

        if z.ndim == 5 and z.shape[2] == 1:
            z = z.squeeze(2)

        # Match Show-o2 visual modules' dtype/device
        visual_dtype = model.image_embedder_und.proj.weight.dtype
        visual_device = model.image_embedder_und.proj.weight.device

        z = z.to(
            device=visual_device,
            dtype=visual_dtype,
        )

        und = model.image_embedder_und(z)
        gen = model.image_embedder_gen(z)

        und = (
            und
            + model.position_embedding(
                model.image_position_ids
            )
        )

        und = model.und_trans(
            und
        )["last_hidden_state"]

    # IMPORTANT:
    # fusion_proj is OUTSIDE torch.no_grad().
    # Therefore this module receives gradients.
    fused = model.fusion_proj(
        torch.cat(
            [und, gen],
            dim=-1,
        )
    )

    return fused


# ============================================================
# TOKEN GRID ALIGNMENT
# ============================================================

def spatial_align_showo_to_qwen(
    showo_features,
    qwen_features,
):

    """
    Show-o2:
        27 x 27 = 729 visual tokens

    Qwen3-VL at 448x448:
        14 x 14 = 196 visual tokens

    No learnable layer is used here.
    Adaptive average pooling only.
    """

    B, Ns, Ds = showo_features.shape
    _, Nq, _ = qwen_features.shape

    s_side = int(math.sqrt(Ns))
    q_side = int(math.sqrt(Nq))

    assert s_side * s_side == Ns, Ns
    assert q_side * q_side == Nq, Nq

    x = showo_features.transpose(1, 2)

    x = x.reshape(
        B,
        Ds,
        s_side,
        s_side,
    )

    x = F.adaptive_avg_pool2d(
        x,
        (q_side, q_side),
    )

    x = x.flatten(2).transpose(1, 2)

    return x


# ============================================================
# LOSS
# ============================================================

def alignment_loss(pred, target):

    pred_fp32 = pred.float()
    target_fp32 = target.float()

    mse = F.mse_loss(
        pred_fp32,
        target_fp32,
    )

    cosine = (
        1.0
        - F.cosine_similarity(
            pred_fp32,
            target_fp32,
            dim=-1,
        ).mean()
    )

    loss = mse + 0.1 * cosine

    return loss, mse, cosine

@torch.no_grad()
def validate(
    val_loader,
    qwen_visual,
    qwen_processor,
    showo2,
    vae,
    adapter,
):
    adapter.eval()
    showo2.fusion_proj.eval()

    total_loss = 0.0
    total_mse = 0.0
    total_cosine = 0.0
    total_samples = 0

    pbar = tqdm(
        val_loader,
        desc="Validation",
        leave=False,
    )

    for batch in pbar:

        images = []

        for item in batch:
            with Image.open(item["path"]) as img:
                images.append(img.convert("RGB"))

        # ----------------------------------------
        # Frozen Qwen3-VL target features
        # ----------------------------------------
        z_qwen = get_qwen_features(
            images,
            qwen_visual,
            qwen_processor,
        )

        # ----------------------------------------
        # Show-o2 features
        # fusion_proj is evaluated, not trained
        # ----------------------------------------
        z_showo = get_showo2_features(
            images,
            showo2,
            vae,
        )

        # Match Show-o2 token grid to Qwen grid
        z_showo = spatial_align_showo_to_qwen(
            z_showo,
            z_qwen,
        )

        # 1536 -> 2560
        z_pred = adapter(z_showo)

        loss, mse, cosine = alignment_loss(
            z_pred,
            z_qwen,
        )

        batch_size = len(batch)

        total_loss += loss.item() * batch_size
        total_mse += mse.item() * batch_size
        total_cosine += cosine.item() * batch_size
        total_samples += batch_size

        pbar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    avg_loss = total_loss / total_samples
    avg_mse = total_mse / total_samples
    avg_cosine = total_cosine / total_samples

    return avg_loss, avg_mse, avg_cosine
# ============================================================
# MAIN
# ============================================================

def main():

    torch.manual_seed(SEED)
    random.seed(SEED)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    dataset = DenseFusionDataset(
        DENSEFUSION_JSON,
        IMAGE_ROOT,
    )

    # Deterministic split.
    indices = list(range(len(dataset)))

    random.Random(SEED).shuffle(indices)

    val_size = int(
        len(indices) * VAL_FRACTION
    )

    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_dataset = torch.utils.data.Subset(
        dataset,
        train_indices,
    )

    val_dataset = torch.utils.data.Subset(
        dataset,
        val_indices,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
    )

    # --------------------------------------------------------
    # MODELS
    # --------------------------------------------------------

    qwen_visual, qwen_processor = (
        load_qwen_visual()
    )

    showo2, vae = load_showo2_visual()

    adapter = Showo2ToQwenAdapter().to(
        DEVICE,
        dtype=DTYPE,
    )

    optimizer = torch.optim.AdamW(
        [
            {
                "params":
                    adapter.parameters(),
                "lr": LR_ADAPTER,
            },
            {
                "params":
                    showo2.fusion_proj.parameters(),
                "lr": LR_FUSION,
            },
        ],
        weight_decay=0.01,
    )

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    global_step = 0

    best_val_loss = float("inf")
    best_epoch = -1

    for epoch in range(EPOCHS):

        adapter.train()
        showo2.fusion_proj.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{EPOCHS}"
        )

        for step, batch in enumerate(pbar):

            images = [
                Image.open(item["path"])
                .convert("RGB")
                for item in batch
            ]

            # Frozen Qwen target representation.
            with torch.no_grad():
                z_qwen = get_qwen_features(
                    images,
                    qwen_visual,
                    qwen_processor,
                )

            # Show-o2 visual representation.
            z_showo = get_showo2_features(
                images,
                showo2,
                vae,
            )

            # 729 tokens -> Qwen token grid.
            z_showo = spatial_align_showo_to_qwen(
                z_showo,
                z_qwen,
            )

            # 1536 -> 2560.
            z_pred = adapter(
                z_showo
            )

            loss, mse, cosine = alignment_loss(
                z_pred,
                z_qwen.detach(),
            )

            (
                loss / GRAD_ACCUM_STEPS
            ).backward()

            if (
                (step + 1)
                % GRAD_ACCUM_STEPS
                == 0
            ):

                torch.nn.utils.clip_grad_norm_(
                    list(adapter.parameters())
                    + list(
                        showo2
                        .fusion_proj
                        .parameters()
                    ),
                    max_norm=1.0,
                )

                optimizer.step()

                optimizer.zero_grad(
                    set_to_none=True
                )

                global_step += 1

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                mse=f"{mse.item():.4f}",
                cos=f"{cosine.item():.4f}",
            )

        print(f"\nRunning validation after epoch {epoch + 1}...")

        val_loss, val_mse, val_cosine = validate(
            val_loader,
            qwen_visual,
            qwen_processor,
            showo2,
            vae,
            adapter,
        )

        print(
            f"Epoch {epoch + 1} validation | "
            f"loss={val_loss:.6f} | "
            f"mse={val_mse:.6f} | "
            f"cosine={val_cosine:.6f}"
        )
        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        checkpoint = {
            "epoch": epoch + 1,
            "global_step": global_step,

            "adapter": adapter.state_dict(),

            "fusion_proj":
                showo2.fusion_proj.state_dict(),

            "optimizer":
                optimizer.state_dict(),

            "val_loss": val_loss,
            "val_mse": val_mse,
            "val_cosine": val_cosine,
        }

        epoch_path = os.path.join(
            OUTPUT_DIR,
            f"checkpoint_epoch_{epoch + 1}.pt",
        )

        torch.save(
            checkpoint,
            epoch_path,
        )

        print(f"Saved: {epoch_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1

            best_path = os.path.join(
                OUTPUT_DIR,
                "checkpoint_best.pt",
            )

            torch.save(
                checkpoint,
                best_path,
            )

            print(
                f"New best checkpoint! "
                f"Epoch {best_epoch}, "
                f"val_loss={best_val_loss:.6f}"
            )

    print("\nTraining finished.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(
        f"Best checkpoint: "
        f"{os.path.join(OUTPUT_DIR, 'checkpoint_best.pt')}"
    )





if __name__ == "__main__":
    main()