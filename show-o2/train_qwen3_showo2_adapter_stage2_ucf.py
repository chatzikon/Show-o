import os
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from tqdm import tqdm

# Reuse the code that already worked in Stage 1
from train_qwen3_showo2_adapter import (
    Showo2ToQwenAdapter,
    load_qwen_visual,
    load_showo2_visual,
    get_qwen_features,
    get_showo2_features,
    spatial_align_showo_to_qwen,
    alignment_loss,
    DEVICE,
    DTYPE,
)


# ============================================================
# CONFIG
# ============================================================

# Change this to the directory where you extracted UCF-Crime images.
# Subdirectories are fine: the dataset searches recursively.
UCF_TRAIN_ROOT = "/home/chatziko/PycharmProjects/PythonProject/IDMVAE/archive/UCA_image_dataset/train/images"
UCF_VAL_ROOT   = "/home/chatziko/PycharmProjects/PythonProject/IDMVAE/archive/UCA_image_dataset/test"

# Best checkpoint from DenseFusion Stage 1
STAGE1_CHECKPOINT = (
    "/home/chatziko/PycharmProjects/PythonProject/Show-o/show-o2/showo2_qwen3_adapter_stage1/checkpoint_best.pt"
)

OUTPUT_DIR = "./showo2_qwen3_adapter_stage2_ucf"

# UCF is tiny, so use a smaller LR than Stage 1.
LR_ADAPTER = 1e-5

BATCH_SIZE = 12

# This is a maximum. Early stopping will normally stop us earlier.
MAX_EPOCHS = 10


EARLY_STOP_PATIENCE = 3

# Ignore extremely tiny fluctuations for early-stopping purposes.
MIN_DELTA = 1e-5

SEED = 42


# ============================================================
# DATASET
# ============================================================

class UCFImageDataset(Dataset):

    EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp",
    }

    def __init__(self, root):
        self.root = Path(root)

        all_paths = sorted(
            p for p in self.root.rglob("*")
            if p.is_file()
            and p.suffix.lower() in self.EXTENSIONS
        )

        if len(all_paths) == 0:
            raise RuntimeError(
                f"No images found under: {self.root}"
            )

        # Since the dataset is small, verify every image once.
        self.paths = []

        for p in all_paths:
            try:
                with Image.open(p) as img:
                    img.verify()

                self.paths.append(str(p))

            except Exception as e:
                print(
                    f"Skipping unreadable image: {p}\n"
                    f"Reason: {e}"
                )

        print(
            f"Found {len(self.paths)} valid UCF images"
        )

        if len(self.paths) == 0:
            raise RuntimeError(
                "No valid UCF images remain."
            )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return {
            "path": self.paths[idx]
        }


def collate_fn(batch):
    return batch


def load_images(batch):

    images = []

    for item in batch:

        with Image.open(item["path"]) as img:
            images.append(
                img.convert("RGB")
            )

    return images


# ============================================================
# VALIDATION
# ============================================================

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
    showo2.eval()

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

        images = load_images(batch)

        # ----------------------------------------
        # Frozen Qwen3-VL representation
        # ----------------------------------------

        z_qwen = get_qwen_features(
            images,
            qwen_visual,
            qwen_processor,
        )

        # ----------------------------------------
        # Frozen Show-o2 representation
        # Including frozen Stage-1 fusion_proj
        # ----------------------------------------

        z_showo = get_showo2_features(
            images,
            showo2,
            vae,
        )

        z_showo = spatial_align_showo_to_qwen(
            z_showo,
            z_qwen,
        )

        # ----------------------------------------
        # Stage-2 trainable mapper
        # ----------------------------------------

        z_pred = adapter(
            z_showo
        )

        loss, mse, cosine = alignment_loss(
            z_pred,
            z_qwen,
        )

        bs = len(batch)

        total_loss += loss.item() * bs
        total_mse += mse.item() * bs
        total_cosine += cosine.item() * bs

        total_samples += bs

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            mse=f"{mse.item():.4f}",
            cos=f"{cosine.item():.4f}",
        )

    return (
        total_loss / total_samples,
        total_mse / total_samples,
        total_cosine / total_samples,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    random.seed(SEED)
    torch.manual_seed(SEED)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    # ========================================================
    # DATASET
    # ========================================================



    train_dataset = UCFImageDataset(
        UCF_TRAIN_ROOT
    )

    val_dataset = UCFImageDataset(
        UCF_VAL_ROOT
    )

    print(
        f"Train images: {len(train_dataset)}"
    )

    print(
        f"Validation images: {len(val_dataset)}"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # ========================================================
    # LOAD FROZEN QWEN3-VL VISUAL ENCODER
    # ========================================================

    print("\nLoading Qwen3-VL visual encoder...")

    qwen_visual, qwen_processor = (
        load_qwen_visual()
    )

    # ========================================================
    # LOAD SHOW-O2 VISUAL PATH
    # ========================================================

    print("\nLoading Show-o2 visual pathway...")

    showo2, vae = (
        load_showo2_visual()
    )

    # ========================================================
    # LOAD STAGE-1 CHECKPOINT
    # ========================================================

    print(
        f"\nLoading Stage-1 checkpoint:\n"
        f"{STAGE1_CHECKPOINT}"
    )

    stage1 = torch.load(
        STAGE1_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    # First restore Stage-1 fusion projection.
    showo2.fusion_proj.load_state_dict(
        stage1["fusion_proj"]
    )

    # Instantiate adapter.
    adapter = Showo2ToQwenAdapter().to(
        device=DEVICE,
        dtype=DTYPE,
    )

    # Restore Stage-1 adapter.
    adapter.load_state_dict(
        stage1["adapter"]
    )

    print(
        f"Loaded Stage-1 epoch: "
        f"{stage1.get('epoch', 'unknown')}"
    )

    print(
        f"Stage-1 validation loss: "
        f"{stage1.get('val_loss', 'unknown')}"
    )

    # ========================================================
    # FREEZE EVERYTHING IN SHOW-O2
    # ========================================================

    for p in showo2.parameters():
        p.requires_grad = False

    showo2.eval()

    # Adapter is the ONLY trainable component.
    for p in adapter.parameters():
        p.requires_grad = True

    adapter.train()

    # ========================================================
    # VERIFY TRAINABLE PARAMETERS
    # ========================================================

    showo_trainable = sum(
        p.numel()
        for p in showo2.parameters()
        if p.requires_grad
    )

    adapter_trainable = sum(
        p.numel()
        for p in adapter.parameters()
        if p.requires_grad
    )

    print("\nStage-2 trainable parameters:")
    print(
        f"Show-o2: {showo_trainable:,}"
    )
    print(
        f"Adapter: {adapter_trainable:,}"
    )

    assert showo_trainable == 0, (
        "ERROR: some Show-o2 parameters "
        "are still trainable."
    )

    # For a single Linear(1536, 2560),
    # this should be ~3.93M parameters.

    # ========================================================
    # NEW STAGE-2 OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=LR_ADAPTER,
        weight_decay=0.01,
    )

    # IMPORTANT:
    # We intentionally DO NOT load the Stage-1 optimizer.
    # Stage 2 begins with a fresh low-LR optimizer.

    # ========================================================
    # TRAINING STATE
    # ========================================================

    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0

    global_step = 0

    # ========================================================
    # TRAIN
    # ========================================================

    for epoch in range(MAX_EPOCHS):

        adapter.train()
        showo2.eval()

        total_train_loss = 0.0
        total_train_samples = 0

        pbar = tqdm(
            train_loader,
            desc=(
                f"Stage 2 Epoch "
                f"{epoch + 1}/{MAX_EPOCHS}"
            ),
        )

        for batch in pbar:

            images = load_images(batch)

            # ----------------------------------------
            # Both visual feature extractors are frozen.
            # ----------------------------------------

            with torch.no_grad():

                z_qwen = get_qwen_features(
                    images,
                    qwen_visual,
                    qwen_processor,
                )

                z_showo = get_showo2_features(
                    images,
                    showo2,
                    vae,
                )

                z_showo = (
                    spatial_align_showo_to_qwen(
                        z_showo,
                        z_qwen,
                    )
                )

            # ----------------------------------------
            # ONLY adapter gets gradients
            # ----------------------------------------

            z_pred = adapter(
                z_showo
            )

            loss, mse, cosine = (
                alignment_loss(
                    z_pred,
                    z_qwen.detach(),
                )
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                adapter.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            global_step += 1

            bs = len(batch)

            total_train_loss += (
                loss.item() * bs
            )

            total_train_samples += bs

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                mse=f"{mse.item():.4f}",
                cos=f"{cosine.item():.4f}",
            )

        avg_train_loss = (
            total_train_loss
            / total_train_samples
        )

        # ====================================================
        # VALIDATION
        # ====================================================

        print(
            f"\nRunning UCF validation "
            f"after epoch {epoch + 1}..."
        )

        (
            val_loss,
            val_mse,
            val_cosine,
        ) = validate(
            val_loader,
            qwen_visual,
            qwen_processor,
            showo2,
            vae,
            adapter,
        )

        print(
            f"Epoch {epoch + 1} | "
            f"train={avg_train_loss:.6f} | "
            f"val={val_loss:.6f} | "
            f"mse={val_mse:.6f} | "
            f"cosine={val_cosine:.6f}"
        )

        # ====================================================
        # SAVE EVERY EPOCH
        # ====================================================

        checkpoint = {

            "stage": 2,

            "epoch":
                epoch + 1,

            "global_step":
                global_step,

            "stage1_checkpoint":
                STAGE1_CHECKPOINT,

            "adapter":
                adapter.state_dict(),

            # unchanged, but storing it makes the checkpoint
            # self-contained for inference.
            "fusion_proj":
                showo2
                .fusion_proj
                .state_dict(),

            "optimizer":
                optimizer.state_dict(),

            "train_loss":
                avg_train_loss,

            "val_loss":
                val_loss,

            "val_mse":
                val_mse,

            "val_cosine":
                val_cosine,
        }

        epoch_path = os.path.join(
            OUTPUT_DIR,
            (
                f"checkpoint_epoch_"
                f"{epoch + 1}.pt"
            ),
        )

        torch.save(
            checkpoint,
            epoch_path,
        )

        print(
            f"Saved: {epoch_path}"
        )

        # ====================================================
        # BEST CHECKPOINT + EARLY STOPPING
        # ====================================================

        if (
            val_loss
            < best_val_loss - MIN_DELTA
        ):

            best_val_loss = val_loss
            best_epoch = epoch + 1

            epochs_without_improvement = 0

            best_path = os.path.join(
                OUTPUT_DIR,
                "checkpoint_best.pt",
            )

            torch.save(
                checkpoint,
                best_path,
            )

            print(
                f"New best UCF checkpoint! "
                f"Epoch {best_epoch}, "
                f"val_loss="
                f"{best_val_loss:.6f}"
            )

        else:

            epochs_without_improvement += 1

            print(
                "No validation improvement "
                f"for "
                f"{epochs_without_improvement}/"
                f"{EARLY_STOP_PATIENCE} "
                "epochs."
            )

        if (
            epochs_without_improvement
            >= EARLY_STOP_PATIENCE
        ):

            print(
                "\nEarly stopping triggered."
            )

            break

    # ========================================================
    # FINISHED
    # ========================================================

    print("\nStage-2 training finished.")

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"Best UCF validation loss: "
        f"{best_val_loss:.6f}"
    )

    print(
        "Best checkpoint: "
        + os.path.join(
            OUTPUT_DIR,
            "checkpoint_best.pt",
        )
    )


if __name__ == "__main__":
    from fusion_version import run_fusion_version

    run_fusion_version(
        v1_main=main,
        v2_module="train_qwen3_showo2_deepstack",
        stage=2,
    )