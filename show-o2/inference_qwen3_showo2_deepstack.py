"""Single-image/folder V2 alpha sweep with trained DeepStack heads."""
import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

import train_qwen3_showo2_adapter as base
from inference_qwen3_showo2_fusion_ablation_batch import (
    prepare_qwen_inputs, move_inputs_to_device, resolve_images, resolve_prompts,
)
from qwen3_showo2_deepstack import load_adapter, unpack_visual, fuse_features, patched_image_features


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--prompt-file", required=True)
    p.add_argument("--output", required=True, help="Fresh output directory for V2 JSON results")
    p.add_argument("--vae-path", default=base.WAN_VAE_PATH)
    p.add_argument("--alphas", nargs="+", type=float, default=[i / 10 for i in range(11)])
    p.add_argument("--max-new-tokens", type=int, default=512)
    args = p.parse_args()
    if any(not 0 <= a <= 1 for a in args.alphas):
        p.error("alphas must be in [0, 1]")
    if args.max_new_tokens < 1:
        p.error("max-new-tokens must be positive")
    base.WAN_VAE_PATH = args.vae_path
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    adapter = load_adapter(checkpoint).eval()  # Reject V1 before loading large models.
    if checkpoint["qwen_model"] != base.QWEN3_MODEL:
        raise ValueError("Checkpoint Qwen model differs from configured model")
    image_paths, prompt_paths = resolve_images(args.image), resolve_prompts(args.prompt_file)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError("Use an empty V2 output directory; existing results will not be overwritten")
    for image_path in image_paths:
        with Image.open(image_path) as im:
            image = im.convert("RGB")
        # Sequential loading follows the existing memory-conscious inference path.
        showo, vae = base.load_showo2_visual()
        showo.fusion_proj.load_state_dict(checkpoint["fusion_proj"])
        showo.eval()
        raw = base.get_showo2_features([image], showo, vae, deterministic_vae=True).cpu()
        del showo, vae
        gc.collect()
        torch.cuda.empty_cache()
        processor = AutoProcessor.from_pretrained(base.QWEN3_MODEL)
        qwen = Qwen3VLForConditionalGeneration.from_pretrained(
            base.QWEN3_MODEL, dtype=base.DTYPE, low_cpu_mem_usage=True,
            attn_implementation="sdpa").to(base.DEVICE).eval()
        if list(qwen.model.visual.deepstack_visual_indexes) != checkpoint["deepstack_indexes"]:
            raise ValueError("Checkpoint DeepStack layer indexes differ from Qwen")
        first = move_inputs_to_device(prepare_qwen_inputs(processor, image, "Describe this image."), base.DEVICE)
        embeddings, deep = unpack_visual(qwen.model.get_image_features(
            pixel_values=first["pixel_values"], image_grid_thw=first["image_grid_thw"]))
        if not isinstance(embeddings, (tuple, list)) or len(embeddings) != 1:
            raise ValueError("Expected one image's split final embeddings (Transformers 4.57.6)")
        native = embeddings[0], deep
        aligned = base.spatial_align_showo_to_qwen(raw.to(native[0]), native[0].unsqueeze(0))
        adapter.to(device=base.DEVICE, dtype=base.DTYPE)
        final, predicted_deep = adapter(aligned)
        adapted = final[0], [d[0] for d in predicted_deep]
        adapter.cpu()
        del first, aligned, raw
        for prompt_path in prompt_paths:
            prompt = Path(prompt_path).read_text(encoding="utf-8")
            inputs = move_inputs_to_device(prepare_qwen_inputs(processor, image, prompt), base.DEVICE)
            results = []
            for alpha in args.alphas:
                final, fused_deep = fuse_features(native, adapted, alpha)
                with patched_image_features(qwen, final, fused_deep):
                    ids = qwen.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True)
                text = processor.batch_decode(ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
                                              clean_up_tokenization_spaces=False)[0]
                results.append(dict(alpha_qwen=alpha, beta_showo2=1-alpha, output=text))
                print(f"{image_path.name} | alpha={alpha}: {text}", flush=True)
            key = hashlib.sha256(f"{image_path}::{prompt_path}".encode()).hexdigest()[:16]
            result = dict(fusion_version=2, fusion_scope="final_and_deepstack", image=str(image_path),
                          prompt_file=str(prompt_path), prompt=prompt, checkpoint=str(Path(args.checkpoint).resolve()),
                          deepstack_indexes=checkpoint["deepstack_indexes"], results=results)
            (out / f"{image_path.stem}__{key}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        del qwen, processor, inputs, native, adapted, embeddings, deep, predicted_deep, final, fused_deep, ids
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()