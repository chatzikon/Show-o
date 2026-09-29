#!/usr/bin/env python3

import argparse
import copy
import gc
import hashlib
import json
import os
import random
import re
import types
from pathlib import Path

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

DEFAULT_IMAGE = (
    "/home/chatziko/PycharmProjects/PythonProject/"
    "test_images_showo2/ucf1.png"
)

DEFAULT_PROMPT_FILE = (
    "/home/chatziko/PycharmProjects/PythonProject/Show-o/show-o2/"
    "prompts/mmu_prompt.txt"
)

DEFAULT_OUTPUT = (
    "/home/chatziko/PycharmProjects/PythonProject/Show-o/show-o2/"
    "results/showo_outputs"
)

MAX_NEW_TOKENS = 512

SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# UTILITIES
# ============================================================

def reset_seed():
    random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


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


def sanitize_name(value):
    """
    Make a filesystem-safe short name.
    """

    value = str(value)

    value = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        value,
    )

    value = value.strip("._-")

    return value or "item"


def short_hash(value):
    return hashlib.sha1(
        str(value).encode("utf-8")
    ).hexdigest()[:8]


def result_json_is_complete(json_path):
    """
    True only for an existing, parseable result JSON containing
    one entry for every configured alpha.
    """
    json_path = Path(json_path)

    if not json_path.is_file():
        return False

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False

    results = data.get("results")

    if not isinstance(results, list):
        return False

    if len(results) != len(ALPHAS):
        return False

    stored_alphas = []

    for result in results:
        if not isinstance(result, dict):
            return False

        alpha = result.get(
            "alpha_qwen",
            result.get("alpha"),
        )

        if alpha is None:
            return False

        try:
            stored_alphas.append(round(float(alpha), 8))
        except (TypeError, ValueError):
            return False

    expected_alphas = [
        round(float(alpha), 8)
        for alpha in ALPHAS
    ]

    return stored_alphas == expected_alphas



def resolve_images(image_arg):
    """
    --image may point to:
      1) one image file
      2) a directory containing images

    Directory mode is recursive.
    """

    path = Path(image_arg).expanduser()

    if not path.exists():
        raise FileNotFoundError(
            f"Image path does not exist: {path}"
        )

    if path.is_file():

        if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            raise ValueError(
                f"Unsupported image extension: {path.suffix}"
            )

        return [path.resolve()]

    images = sorted(
        p.resolve()
        for p in path.rglob("*")
        if (
            p.is_file()
            and
            p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        )
    )

    if not images:
        raise ValueError(
            f"No supported images found in folder: {path}"
        )

    return images


def resolve_prompts(prompt_arg):
    """
    --prompt-file may point to:
      1) one .txt prompt
      2) a directory containing .txt prompts

    Directory mode is recursive.
    """

    path = Path(prompt_arg).expanduser()

    if not path.exists():
        raise FileNotFoundError(
            f"Prompt path does not exist: {path}"
        )

    if path.is_file():
        return [path.resolve()]

    prompts = sorted(
        p.resolve()
        for p in path.rglob("*.txt")
        if p.is_file()
    )

    if not prompts:
        raise ValueError(
            f"No .txt prompt files found in folder: {path}"
        )

    return prompts


def resolve_stages(stage_args):
    """
    Resolve one or more requested stages.

    Examples:
        --stage stage1
        --stage stage1 stage2
        --stage pre_stage1 stage1 stage2
        --stage all
    """

    # Keep backward compatibility in case this function is called
    # programmatically with a single string.
    if isinstance(stage_args, str):
        stage_args = [stage_args]

    if "all" in stage_args:
        if len(stage_args) != 1:
            raise ValueError(
                "'all' cannot be combined with individual stages. "
                "Use either '--stage all' or, for example, "
                "'--stage stage1 stage2'."
            )

        return [
            "pre_stage1",
            "stage1",
            "stage2",
        ]

    # Remove duplicates while preserving the user's requested order.
    stages = []

    for stage in stage_args:
        if stage not in stages:
            stages.append(stage)

    return stages


def is_single_run(
    image_paths,
    prompt_paths,
    stages,
):
    return (
        len(image_paths) == 1
        and
        len(prompt_paths) == 1
        and
        len(stages) == 1
    )


def build_output_paths(
    output_arg,
    image_path,
    prompt_path,
    stage,
    single_run,
):
    """
    Backward-compatible behavior:

    SINGLE combination:
        --output is treated as the exact JSON output path,
        just like the original script.

    MULTIPLE combinations:
        --output is treated as a directory and files are saved as:

        output/
            pre_stage1/
            stage1/
            stage2/

        with one JSON + TXT pair per image/prompt combination.
    """

    output_path = Path(output_arg).expanduser()

    if single_run:

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        json_path = output_path

        txt_path = Path(
            os.path.splitext(
                str(output_path)
            )[0]
            +
            ".txt"
        )

        return json_path, txt_path

    stage_dir = (
        output_path
        /
        stage
    )

    stage_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    image_stem = sanitize_name(
        image_path.stem
    )

    prompt_stem = sanitize_name(
        prompt_path.stem
    )

    # Hash the full paths so identically named images/prompts
    # from different subfolders cannot overwrite one another.
    suffix = short_hash(
        f"{image_path}::{prompt_path}"
    )

    filename = (
        f"{image_stem}"
        f"__{prompt_stem}"
        f"__{suffix}"
    )

    json_path = (
        stage_dir
        /
        f"{filename}.json"
    )

    txt_path = (
        stage_dir
        /
        f"{filename}.txt"
    )

    return json_path, txt_path


# ============================================================
# LOAD SHOW-O2 REPRESENTATION
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
# PREPARE A STAGE REPRESENTATION
# ============================================================

def prepare_stage_representation(
    stage,
    stage_data,
    z_qwen,
):
    """
    Create the stage-specific adapted Show-o2 representation.

    stage_data contains:
        z_showo_raw
        adapter_state
        checkpoint_path

    Qwen is already loaded at this point.
    """

    z_showo_raw = stage_data["z_showo_raw"]
    adapter_state = stage_data["adapter_state"]

    # --------------------------------------------------------
    # Prepare adapter
    # --------------------------------------------------------

    if stage == "pre_stage1":

        # Reset RNG immediately before initialization so that
        # the random adapter is identical across runs.
        reset_seed()

    adapter = Showo2ToQwenAdapter().to(
        device=DEVICE,
        dtype=DTYPE,
    )

    if adapter_state is not None:

        adapter.load_state_dict(
            adapter_state
        )

        print(
            f"Loaded trained adapter for {stage}"
        )

    else:

        print(
            "Using RANDOMLY INITIALIZED adapter "
            "(pre-Stage1 baseline)"
        )

    adapter.eval()

    for p in adapter.parameters():
        p.requires_grad = False

    # --------------------------------------------------------
    # Match Show-o2 spatial grid to Qwen visual grid
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Map 1536 -> 2560
    # --------------------------------------------------------

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
        ==
        z_showo_adapted.shape
    ), (
        f"Shape mismatch: "
        f"Qwen={z_qwen.shape}, "
        f"Show-o2={z_showo_adapted.shape}"
    )

    # --------------------------------------------------------
    # Feature norm check
    # --------------------------------------------------------

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
    print(f"Stage: {stage}")
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

    del adapter
    del z_showo_aligned

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return (
        z_showo_adapted,
        qwen_norm,
        showo_norm,
    )


# ============================================================
# GENERATE ALL ALPHAS FOR ONE STAGE + PROMPT
# ============================================================

@torch.no_grad()
def generate_alpha_sweep(
    qwen,
    processor,
    inputs,
    native_outputs,
    z_qwen,
    z_showo_adapted,
    qwen_norm,
    showo_norm,
    max_new_tokens,
):
    """
    ALPHAS are already defined globally and are swept exactly
    as in the original script.
    """

    native_image_embeds, native_deepstack = native_outputs

    original_get_image_features = (
        qwen.model.get_image_features
    )

    results = []

    try:

        for alpha in ALPHAS:

            beta = 1.0 - alpha

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

            # ------------------------------------------------
            # Patch Qwen's image-feature function.
            #
            # Replace ONLY its final visual embeddings.
            # DeepStack remains native Qwen.
            # ------------------------------------------------

            def fused_get_image_features(
                    self,
                    pixel_values,
                    image_grid_thw=None,
                    **kwargs,
            ):
                # One image:
                # fused has shape [1, N, D],
                # Qwen expects a tuple containing [N, D].
                fused_image_embeds = (
                    fused[0],
                )

                # Keep native Qwen DeepStack features.
                return (
                    fused_image_embeds,
                    native_deepstack,
                )

            qwen.model.get_image_features = (
                types.MethodType(
                    fused_get_image_features,
                    qwen.model,
                )
            )

            # Important when repeatedly generating using
            # different image embeddings.
            qwen.model.rope_deltas = None

            generated_ids = qwen.generate(
                **inputs,

                max_new_tokens=
                    max_new_tokens,

                do_sample=False,

                use_cache=True,
            )

            # Strip original prompt tokens.
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

    finally:

        # Always restore the native method even if generation
        # fails part-way through the alpha sweep.
        qwen.model.get_image_features = (
            original_get_image_features
        )

        qwen.model.rope_deltas = None

    return results


# ============================================================
# SAVE ONE RESULT
# ============================================================

def save_result(
    json_path,
    txt_path,
    image_path,
    stage,
    checkpoint_path,
    prompt_path,
    prompt,
    qwen_norm,
    showo_norm,
    results,
):

    output = {
        "image": str(image_path),

        "stage":
            stage,

        "checkpoint":
            checkpoint_path,

        "adapter_status":
            (
                "random_initialization"
                if stage == "pre_stage1"
                else "trained"
            ),

        "prompt_file":
            str(prompt_path),

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

    json_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False,
        )

    txt_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        txt_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "SHOW-O2 + QWEN3-VL FUSION ABLATION\n"
        )

        f.write(
            "=" * 60
            +
            "\n\n"
        )

        f.write(
            f"Image: {image_path}\n"
        )

        f.write(
            f"Stage: {stage}\n"
        )

        f.write(
            f"Checkpoint: {checkpoint_path}\n"
        )

        if stage == "pre_stage1":

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
            f"Qwen mean token norm: "
            f"{qwen_norm:.6f}\n"
        )

        f.write(
            f"Show-o2 mean token norm: "
            f"{showo_norm:.6f}\n"
        )

        f.write(
            "Show-o2 / Qwen ratio: "
            f"{showo_norm / qwen_norm:.6f}\n\n"
        )

        f.write("RESULTS\n")
        f.write("=" * 60 + "\n\n")

        for result in results:

            f.write(
                f"alpha="
                f"{result['alpha_qwen']:.1f} "
                f"(Qwen="
                f"{result['alpha_qwen']:.1f}, "
                f"Show-o2="
                f"{result['beta_showo2']:.1f})\n"
            )

            f.write("-" * 60 + "\n")

            f.write(
                result["output"].strip()
            )

            f.write("\n\n")

    print(
        f"JSON results saved to: "
        f"{json_path}"
    )

    print(
        f"Text results saved to: "
        f"{txt_path}"
    )


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_image(
    image_path,
    prompt_paths,
    stages,
    args,
    single_run,
):

    print("\n\n")
    print("#" * 72)
    print(f"IMAGE: {image_path}")
    print("#" * 72)

    # ========================================================
    # STEP 0:
    # Determine which prompt × stage jobs still need to run.
    # ========================================================

    pending_jobs = []

    for prompt_path in prompt_paths:
        for stage in stages:

            json_path, txt_path = build_output_paths(
                output_arg=args.output,
                image_path=image_path,
                prompt_path=prompt_path,
                stage=stage,
                single_run=single_run,
            )

            if (
                not args.overwrite_existing
                and result_json_is_complete(json_path)
            ):
                print(
                    "[SKIP] Existing complete result: "
                    f"{json_path}"
                )
                continue

            pending_jobs.append(
                {
                    "prompt_path": prompt_path,
                    "stage": stage,
                    "json_path": json_path,
                    "txt_path": txt_path,
                }
            )

    if not pending_jobs:
        print(
            "[SKIP IMAGE] All requested results already exist "
            "and are complete."
        )
        return

    active_stages = [
        stage
        for stage in stages
        if any(
            job["stage"] == stage
            for job in pending_jobs
        )
    ]

    active_prompt_paths = [
        prompt_path
        for prompt_path in prompt_paths
        if any(
            job["prompt_path"] == prompt_path
            for job in pending_jobs
        )
    ]

    pending_lookup = {
        (job["prompt_path"], job["stage"]): job
        for job in pending_jobs
    }

    print(f"Pending jobs for this image: {len(pending_jobs)}")
    print("Pending stages: " + ", ".join(active_stages))
    print(f"Pending prompts: {len(active_prompt_paths)}")

    with Image.open(image_path) as im:
        image = im.convert("RGB")

    # ========================================================
    # STEP 1:
    # Extract Show-o2 representations only for needed stages.
    # ========================================================

    stage_data = {}

    for stage in active_stages:

        (
            z_showo_raw,
            adapter_state,
            checkpoint_path,
        ) = extract_showo_features(
            image,
            stage,
        )

        stage_data[stage] = {
            "z_showo_raw": z_showo_raw,
            "adapter_state": adapter_state,
            "checkpoint_path": checkpoint_path,
        }

    # ========================================================
    # STEP 2: load Qwen once for this image.
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
    # STEP 3: native Qwen visual representation.
    # ========================================================

    first_prompt = read_prompt(
        active_prompt_paths[0]
    )

    first_inputs = prepare_qwen_inputs(
        processor,
        image,
        first_prompt,
    )

    first_inputs = move_inputs_to_device(
        first_inputs,
        DEVICE,
    )

    print("\nExtracting native Qwen visual features...")

    with torch.no_grad():
        native_image_embeds, native_deepstack = qwen.get_image_features(
            pixel_values=first_inputs["pixel_values"],
            image_grid_thw=first_inputs["image_grid_thw"],
        )

    # One image only.
    # Keep batch dimension because the rest of your code expects [1, N, D].
    z_qwen = native_image_embeds[0].unsqueeze(0)

    # Preserve both components for the monkey-patched generation.
    native_outputs = (
        native_image_embeds,
        native_deepstack,
    )

    print(
        "Qwen feature shape:",
        tuple(z_qwen.shape),
    )


    del first_inputs

    # ========================================================
    # STEP 4: prepare only needed stage representations.
    # ========================================================

    stage_ready = {}

    for stage in active_stages:

        print("\n")
        print("*" * 72)
        print(f"PREPARING STAGE: {stage}")
        print("*" * 72)

        (
            z_showo_adapted,
            qwen_norm,
            showo_norm,
        ) = prepare_stage_representation(
            stage,
            stage_data[stage],
            z_qwen,
        )

        stage_ready[stage] = {
            "z_showo_adapted": z_showo_adapted,
            "qwen_norm": qwen_norm,
            "showo_norm": showo_norm,
            "checkpoint_path": stage_data[stage]["checkpoint_path"],
        }

        stage_data[stage]["z_showo_raw"] = None
        stage_data[stage]["adapter_state"] = None

    stage_data.clear()

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ========================================================
    # STEP 5: run only missing prompt × stage combinations.
    # ========================================================

    total_jobs = len(pending_jobs)
    job_index = 0

    for prompt_path in active_prompt_paths:

        prompt_stages = [
            stage
            for stage in active_stages
            if (prompt_path, stage) in pending_lookup
        ]

        if not prompt_stages:
            continue

        prompt = read_prompt(
            prompt_path
        )

        print("\n\n")
        print("=" * 72)
        print(f"PROMPT: {prompt_path}")
        print("=" * 72)

        inputs = prepare_qwen_inputs(
            processor,
            image,
            prompt,
        )

        inputs = move_inputs_to_device(
            inputs,
            DEVICE,
        )

        for stage in prompt_stages:

            job_index += 1
            job = pending_lookup[
                (prompt_path, stage)
            ]

            print("\n")
            print("#" * 72)
            print(f"JOB {job_index}/{total_jobs}")
            print(f"Image : {image_path.name}")
            print(f"Prompt: {prompt_path.name}")
            print(f"Stage : {stage}")
            print("#" * 72)

            ready = stage_ready[stage]

            results = generate_alpha_sweep(
                qwen=qwen,
                processor=processor,
                inputs=inputs,
                native_outputs=native_outputs,
                z_qwen=z_qwen,
                z_showo_adapted=ready["z_showo_adapted"],
                qwen_norm=ready["qwen_norm"],
                showo_norm=ready["showo_norm"],
                max_new_tokens=args.max_new_tokens,
            )

            save_result(
                json_path=job["json_path"],
                txt_path=job["txt_path"],
                image_path=image_path,
                stage=stage,
                checkpoint_path=ready["checkpoint_path"],
                prompt_path=prompt_path,
                prompt=prompt,
                qwen_norm=ready["qwen_norm"],
                showo_norm=ready["showo_norm"],
                results=results,
            )

        del inputs

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del stage_ready
    del native_outputs
    del z_qwen
    del qwen
    del processor
    del image

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Show-o2 + Qwen3-VL fusion ablation. "
            "--image may be one image or an image folder; "
            "--prompt-file may be one prompt or a prompt folder; "
            "--stage may contain one or more stages."
        )
    )

    parser.add_argument(
        "--image",
        #default='/home/chatziko/PycharmProjects/PythonProject/IDMVAE/archive/UCA_image_dataset/one_frame_per_video_split',
        default='/home/chatziko/PycharmProjects/PythonProject/test_images_showo2',
        type=str,
        help=(
            "Path to one image OR a folder containing images. "
            "Folder mode is recursive."
        ),
    )

    parser.add_argument(
        "--prompt-file",
        default='/home/chatziko/PycharmProjects/PythonProject/showo2_cmalign/CMAlign/api',
        type=str,
        help=(
            "Path to one .txt prompt OR a folder containing "
            ".txt prompt files. Folder mode is recursive."
        ),
    )

    parser.add_argument(
        "--output",
        default='/home/chatziko/PycharmProjects/PythonProject/Show-o/show-o2/results/ablation_results',
        type=str,
        help=(
            "Single-run: exact output JSON path, preserving the "
            "original behavior. Multi-run: output directory."
        ),
    )

    parser.add_argument(
        "--max-new-tokens",
        default=MAX_NEW_TOKENS,
        type=int,
    )

    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        default=False,
        help=(
            "Regenerate and overwrite complete existing result JSONs. "
            "Default behavior is to skip them, enabling safe resume."
        ),
    )

    parser.add_argument(
        "--stage",
        nargs="+",
        choices=[
            "pre_stage1",
            "stage1",
            "stage2",
            "all",
        ],
        default=[
            "stage1",
            "stage2",
        ],
        type=str,
        help=(
            "Run one or more stages. Examples: "
            "'--stage stage1', '--stage stage1 stage2', "
            "or '--stage all'. "
            "Default: stage1 stage2."
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # RESOLVE REQUESTED INPUTS
    # ========================================================

    image_paths = resolve_images(
        args.image
    )

    prompt_paths = resolve_prompts(
        args.prompt_file
    )

    stages = resolve_stages(
        args.stage
    )

    single_run = is_single_run(
        image_paths,
        prompt_paths,
        stages,
    )

    print("\n")
    print("=" * 72)
    print("RUN CONFIGURATION")
    print("=" * 72)

    print(
        f"Images : {len(image_paths)}"
    )

    print(
        f"Prompts: {len(prompt_paths)}"
    )

    print(
        "Stages : "
        +
        ", ".join(stages)
    )

    print(
        f"Alphas : {len(ALPHAS)} "
        f"({ALPHAS[0]:.1f} -> "
        f"{ALPHAS[-1]:.1f})"
    )

    total_result_sets = (
        len(image_paths)
        *
        len(prompt_paths)
        *
        len(stages)
    )

    total_generations = (
        total_result_sets
        *
        len(ALPHAS)
    )

    print(
        f"Result sets: "
        f"{total_result_sets}"
    )

    print(
        f"Total generations: "
        f"{total_generations}"
    )

    print(
        f"Output: {args.output}"
    )

    print(
        "Existing results: "
        +
        (
            "OVERWRITE"
            if args.overwrite_existing
            else "SKIP complete JSON files"
        )
    )

    print("=" * 72)

    # ========================================================
    # PROCESS IMAGES
    # ========================================================

    for index, image_path in enumerate(
        image_paths,
        start=1,
    ):

        print("\n\n")
        print("=" * 72)
        print(
            f"IMAGE {index}/{len(image_paths)}"
        )
        print("=" * 72)

        process_image(
            image_path=
                image_path,

            prompt_paths=
                prompt_paths,

            stages=
                stages,

            args=
                args,

            single_run=
                single_run,
        )

    print("\n")
    print("=" * 72)
    print("ALL REQUESTED RUNS FINISHED")
    print("=" * 72)


if __name__ == "__main__":
    main()