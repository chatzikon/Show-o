import argparse
import copy
import gc
import json
import os
import types

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
)

# Reuse the code that already worked during training.
from train_qwen3_showo2_adapter import (
    Showo2ToQwenAdapter,
    load_showo2_visual,
    get_showo2_features,
    spatial_align_showo_to_qwen,
    DEVICE,
    DTYPE,
    QWEN3_MODEL,
    QWEN_SIZE,
)

from datasets.utils import resize_and_pad_image
import random

SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

torch.use_deterministic_algorithms(
    True,
    warn_only=True,
)


# ============================================================
# CONFIG
# ============================================================

ALPHAS = [
    0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
]


DEFAULT_STAGE1_CHECKPOINT = (
    "/home/chatziko/PycharmProjects/PythonProject/Show-o/show-o2/"
    "showo2_qwen3_adapter_stage1/checkpoint_best.pt"
)

DEFAULT_STAGE2_CHECKPOINT = (
    "/home/chatziko/PycharmProjects/PythonProject/Show-o/show-o2/"
    "showo2_qwen3_adapter_stage2_ucf/checkpoint_best.pt"
)

DEFAULT_PROMPT_FILE = "/home/chatziko/PycharmProjects/PythonProject/Show-o/show-o2/prompts/mmu_prompt.txt"

MAX_NEW_TOKENS = 512


# ============================================================
# UTILITIES
# ============================================================

def read_prompt(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        return f.read().strip()


def move_inputs_to_device(inputs, device):

    result = {}

    for key, value in inputs.items():

        if torch.is_tensor(value):
            result[key] = value.to(device)
        else:
            result[key] = value

    return result


# ============================================================
# LOAD STAGE-2 SHOW-O2 REPRESENTATION
# ============================================================

@torch.no_grad()
def extract_showo_features(
    image,
    stage,
):

    print("\n========================================")
    print("Loading Show-o2 visual pathway")
    print(f"Stage: {stage}")
    print("========================================")

    showo2, vae = load_showo2_visual()

    checkpoint = None
    adapter_state = None
    checkpoint_path = None

    # --------------------------------------------------------
    # Select checkpoint
    # --------------------------------------------------------

    if stage == "stage1":
        checkpoint_path = DEFAULT_STAGE1_CHECKPOINT

    elif stage == "stage2":
        checkpoint_path = DEFAULT_STAGE2_CHECKPOINT

    elif stage == "pre_stage1":
        checkpoint_path = None

    else:
        raise ValueError(
            f"Unknown stage: {stage}"
        )

    # --------------------------------------------------------
    # Stage 1 / Stage 2:
    # Restore trained fusion_proj + adapter.
    #
    # Pre-Stage1:
    # Keep ORIGINAL pretrained Show-o2 fusion_proj.
    # Adapter will be randomly initialized later.
    # --------------------------------------------------------

    if checkpoint_path is not None:

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        print(
            "Checkpoint:",
            checkpoint_path,
        )

        print(
            "Checkpoint epoch:",
            checkpoint.get(
                "epoch",
                "unknown",
            ),
        )

        print(
            "Validation loss:",
            checkpoint.get(
                "val_loss",
                "unknown",
            ),
        )

        showo2.fusion_proj.load_state_dict(
            checkpoint["fusion_proj"]
        )

        adapter_state = checkpoint["adapter"]

    else:

        print(
            "Pre-Stage1 mode:"
            " using original pretrained Show-o2 "
            "fusion_proj."
        )

        print(
            "Adapter will remain randomly initialized."
        )

    showo2.eval()

    # --------------------------------------------------------
    # Extract Show-o2 visual representation
    # --------------------------------------------------------

    z_showo = get_showo2_features(
        [image],
        showo2,
        vae,
        deterministic_vae=True,
    )

    z_showo = (
        z_showo
        .detach()
        .float()
        .cpu()
    )

    print(
        "Raw Show-o2 feature shape:",
        tuple(z_showo.shape),
    )

    del showo2
    del vae

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return (
        z_showo,
        adapter_state,
        checkpoint_path,
    )

# ============================================================
# PREPARE QWEN INPUT
# ============================================================

def prepare_qwen_inputs(
    processor,
    image,
    prompt,
):

    # Use exactly the fixed resolution used during Stage 1.
    qwen_image = resize_and_pad_image(
        image,
        (QWEN_SIZE, QWEN_SIZE),
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Important:
    # image already resized/padded to QWEN_SIZE,
    # so do not resize it again.
    inputs = processor(
        text=[text],
        images=[qwen_image],
        return_tensors="pt",
        do_resize=False,
    )

    return inputs


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        default='/home/chatziko/PycharmProjects/PythonProject/test_images_showo2/ucf1.png',
        type=str,
    )

    parser.add_argument(
        "--prompt-file",
        default=DEFAULT_PROMPT_FILE,
        type=str,
    )

    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_STAGE2_CHECKPOINT,
        type=str,
    )

    parser.add_argument(
        "--output",
        default="/home/chatziko/PycharmProjects/PythonProject/Show-o/show-o2/results/showo_outputs",
        type=str,
    )

    parser.add_argument(
        "--max-new-tokens",
        default=MAX_NEW_TOKENS,
        type=int,
    )
    parser.add_argument(
        "--stage",
        choices=[
            "pre_stage1",
            "stage1",
            "stage2",
        ],
        default="stage1",
        type=str,
    )

    args = parser.parse_args()

    # ========================================================
    # INPUT IMAGE / PROMPT
    # ========================================================

    image = Image.open(
        args.image
    ).convert("RGB")

    prompt = read_prompt(
        args.prompt_file
    )

    # ========================================================
    # STEP 1:
    # Extract Stage-2 Show-o2 representation first.
    #
    # This allows us to delete Show-o2 + Wan VAE BEFORE
    # loading the complete Qwen3-VL-4B model.
    #
    # Important for your 16 GB GPU.
    # ========================================================

    (
        z_showo_raw,
        adapter_state,
        checkpoint_path,
    ) = extract_showo_features(
        image,
        args.stage,
    )

    # ========================================================
    # STEP 2:
    # Load complete Qwen3-VL
    # ========================================================

    print("\n========================================")
    print("Loading full Qwen3-VL")
    print("========================================")

    processor = AutoProcessor.from_pretrained(
        QWEN3_MODEL
    )

    qwen = (
        Qwen3VLForConditionalGeneration
        .from_pretrained(
            QWEN3_MODEL,
            dtype=DTYPE,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        .to(DEVICE)
    )

    qwen.eval()

    for p in qwen.parameters():
        p.requires_grad = False

    # ========================================================
    # STEP 3:
    # Prepare exactly one set of multimodal inputs
    # ========================================================

    inputs = prepare_qwen_inputs(
        processor,
        image,
        prompt,
    )

    inputs = move_inputs_to_device(
        inputs,
        DEVICE,
    )

    # ========================================================
    # STEP 4:
    # Compute native Qwen visual representation ONCE
    # ========================================================

    print("\nExtracting native Qwen visual features...")

    with torch.no_grad():

        native_outputs = (
            qwen.get_image_features(
                pixel_values=
                    inputs["pixel_values"],

                image_grid_thw=
                    inputs["image_grid_thw"],

                return_dict=True,
            )
        )

    # One image only.
    z_qwen = (
        native_outputs
        .pooler_output[0]
        .unsqueeze(0)
    )

    print(
        "Qwen feature shape:",
        tuple(z_qwen.shape),
    )

    # ========================================================
    # STEP 5:
    # Restore Stage-2 adapter
    # ========================================================

    # ========================================================
    # STEP 5:
    # Prepare adapter
    # ========================================================

    if args.stage == "pre_stage1":

        # Reset RNG immediately before initialization so that
        # the random adapter is identical across runs.
        random.seed(SEED)
        torch.manual_seed(SEED)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

    adapter = Showo2ToQwenAdapter().to(
        device=DEVICE,
        dtype=DTYPE,
    )

    if adapter_state is not None:

        adapter.load_state_dict(
            adapter_state
        )

        print(
            f"Loaded trained adapter for {args.stage}"
        )

    else:

        print(
            "Using RANDOMLY INITIALIZED adapter "
            "(pre-Stage1 baseline)"
        )

    adapter.eval()

    for p in adapter.parameters():
        p.requires_grad = False

    adapter.eval()

    for p in adapter.parameters():
        p.requires_grad = False

    # ========================================================
    # STEP 6:
    # Match Show-o2 spatial grid to Qwen visual grid
    #
    # e.g.
    #
    #     Show-o2: 729 = 27 x 27
    #
    #         ↓
    #
    #     Qwen:     196 = 14 x 14
    #
    # ========================================================

    z_showo_raw = z_showo_raw.to(
        device=DEVICE,
        dtype=DTYPE,
    )

    z_showo_aligned = (
        spatial_align_showo_to_qwen(
            z_showo_raw,
            z_qwen,
        )
    )

    print(
        "Spatially aligned Show-o2 shape:",
        tuple(z_showo_aligned.shape),
    )

    # ========================================================
    # STEP 7:
    # Map 1536 -> 2560
    # ========================================================

    with torch.no_grad():

        z_showo_adapted = adapter(
            z_showo_aligned
        )

    print(
        "Adapted Show-o2 shape:",
        tuple(z_showo_adapted.shape),
    )

    assert (
        z_qwen.shape
        == z_showo_adapted.shape
    ), (
        f"Shape mismatch: "
        f"Qwen={z_qwen.shape}, "
        f"Show-o2={z_showo_adapted.shape}"
    )

    # ========================================================
    # FEATURE-NORM CHECK
    # ========================================================

    qwen_norm = (
        z_qwen
        .float()
        .norm(dim=-1)
        .mean()
        .item()
    )

    showo_norm = (
        z_showo_adapted
        .float()
        .norm(dim=-1)
        .mean()
        .item()
    )

    print("\n========================================")
    print("FEATURE NORM CHECK")
    print("========================================")

    print(
        f"Qwen mean token norm: "
        f"{qwen_norm:.6f}"
    )

    print(
        f"Show-o2 mean token norm: "
        f"{showo_norm:.6f}"
    )

    print(
        f"Show-o2 / Qwen norm ratio: "
        f"{showo_norm / qwen_norm:.6f}"
    )

    # ========================================================
    # Preserve original Qwen method
    # ========================================================

    original_get_image_features = (
        qwen.model.get_image_features
    )

    results = []

    # ========================================================
    # STEP 8:
    # WEIGHT SWEEP
    # ========================================================

    for alpha in ALPHAS:

        beta = 1.0 - alpha

        # ----------------------------------------------------
        # Fusion
        #
        # alpha = Qwen weight
        # beta  = Show-o2 weight
        # ----------------------------------------------------

        fused = (
            alpha * z_qwen
            +
            beta * z_showo_adapted
        )

        fused = fused.to(
            dtype=z_qwen.dtype,
            device=z_qwen.device,
        )

        print("\n")
        print("========================================")
        print(
            f"alpha={alpha:.1f} "
            f"(Qwen={alpha:.1f}, "
            f"Show-o2={beta:.1f})"
        )
        print("========================================")

        # ----------------------------------------------------
        # Patch Qwen's image-feature function.
        #
        # We replace ONLY its final visual embeddings.
        #
        # DeepStack features remain the original Qwen3-VL
        # features.
        # ----------------------------------------------------

        def fused_get_image_features(
            self,
            pixel_values,
            image_grid_thw=None,
            **kwargs,
        ):

            output = copy.copy(
                native_outputs
            )

            # Qwen expects a tuple, one tensor per image.
            output.pooler_output = (
                fused[0],
            )

            # output.deepstack_features is unchanged.
            return output

        qwen.model.get_image_features = (
            types.MethodType(
                fused_get_image_features,
                qwen.model,
            )
        )

        # Important when repeatedly generating using
        # different image embeddings.
        qwen.model.rope_deltas = None

        # ----------------------------------------------------
        # Generate deterministically
        # ----------------------------------------------------

        with torch.no_grad():

            generated_ids = qwen.generate(
                **inputs,

                max_new_tokens=
                    args.max_new_tokens,

                do_sample=False,

                use_cache=True,
            )

        # Strip the original prompt tokens.
        generated_only = generated_ids[
            :,
            inputs["input_ids"].shape[1]:
        ]

        output_text = (
            processor.batch_decode(
                generated_only,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        )

        print(output_text)

        result = {
            "alpha_qwen": alpha,
            "beta_showo2": beta,
            "qwen_norm": qwen_norm,
            "showo2_norm": showo_norm,
            "output": output_text,
        }

        results.append(
            result
        )

    # ========================================================
    # RESTORE ORIGINAL QWEN METHOD
    # ========================================================

    qwen.model.get_image_features = (
        original_get_image_features
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    output = {
        "image": args.image,

        "stage":
            args.stage,

        "checkpoint":
            checkpoint_path,

        "adapter_status":
            (
                "random_initialization"
                if args.stage == "pre_stage1"
                else "trained"
            ),

        "prompt_file":
            args.prompt_file,

        "prompt":
            prompt,

        "fusion_formula":
            (
                "alpha * Qwen3-VL + "
                "(1-alpha) * adapted Show-o2"
            ),

        "qwen_mean_token_norm":
            qwen_norm,

        "showo2_mean_token_norm":
            showo_norm,

        "showo2_qwen_norm_ratio":
            showo_norm / qwen_norm,

        "results":
            results,
    }

    with open(
        args.output,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n========================================")
    print("FINISHED")
    print("========================================")

    print(
        f"Results saved to: "
        f"{args.output}"
    )

    txt_path = os.path.splitext(args.output)[0] + ".txt"

    with open(
            txt_path,
            "w",
            encoding="utf-8",
    ) as f:

        f.write("SHOW-O2 + QWEN3-VL FUSION ABLATION\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Image: {args.image}\n")
        f.write(f"Stage: {args.stage}\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")

        if args.stage == "pre_stage1":
            f.write(
                "Adapter: random initialization "
                f"(seed={SEED})\n"
            )

        f.write("\n")

        f.write("PROMPT\n")
        f.write("=" * 60 + "\n")
        f.write(prompt)
        f.write("\n\n")

        f.write("FEATURE NORMS\n")
        f.write("=" * 60 + "\n")
        f.write(
            f"Qwen mean token norm: {qwen_norm:.6f}\n"
        )
        f.write(
            f"Show-o2 mean token norm: {showo_norm:.6f}\n"
        )
        f.write(
            "Show-o2 / Qwen ratio: "
            f"{showo_norm / qwen_norm:.6f}\n\n"
        )

        f.write("RESULTS\n")
        f.write("=" * 60 + "\n\n")

        for result in results:
            f.write(
                f"alpha={result['alpha_qwen']:.1f} "
                f"(Qwen={result['alpha_qwen']:.1f}, "
                f"Show-o2={result['beta_showo2']:.1f})\n"
            )

            f.write("-" * 60 + "\n")

            f.write(
                result["output"].strip()
            )

            f.write("\n\n")

    print(f"JSON results saved to: {args.output}")
    print(f"Text results saved to: {txt_path}")


if __name__ == "__main__":
    main()