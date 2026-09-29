# coding=utf-8
# Copyright 2025 NUS Show Lab.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"

from PIL import Image
import torch
import wandb
from tqdm import tqdm
from accelerate.logging import get_logger

from models import Showo2Qwen2_5, omni_attn_mask_naive
from models.misc import get_text_tokenizer
from utils import (
    get_config,
    flatten_omega_conf,
    get_hyper_params,
    path_to_llm_name,
    load_state_dict,
)
from datasets.utils import image_transform


logger = get_logger(__name__, log_level="INFO")


if __name__ == "__main__":

    config = get_config()

    # -------------------------------------------------------------------------
    # W&B
    # -------------------------------------------------------------------------
    resume_wandb_run = config.wandb.resume
    run_id = config.wandb.get("run_id", None)
    if run_id is None:
        resume_wandb_run = False
        run_id = wandb.util.generate_id()
        config.wandb.run_id = run_id

    wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}

    wandb.init(
        project="demo",
        name=config.experiment.name,
        config=wandb_config,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # FP16 and BF16 use the same amount of memory.
    # Keep one dtype consistently for VAE + Show-o2 to avoid an extra conversion.
    weight_type = torch.float16

    # -------------------------------------------------------------------------
    # Tokenizer / prompt setup
    # -------------------------------------------------------------------------
    text_tokenizer, showo_token_ids = get_text_tokenizer(
        config.model.showo.llm_model_path,
        add_showo_tokens=True,
        return_showo_token_ids=True,
        llm_name=path_to_llm_name[config.model.showo.llm_model_path],
    )
    config.model.showo.llm_vocab_size = len(text_tokenizer)

    if config.model.showo.add_time_embeds:
        # We prepend the time embedding to vision tokens.
        config.dataset.preprocessing.num_t2i_image_tokens += 1
        config.dataset.preprocessing.num_mmu_image_tokens += 1
        config.dataset.preprocessing.num_video_tokens += 1

    (
        num_t2i_image_tokens,
        num_mmu_image_tokens,
        num_video_tokens,
        max_seq_len,
        max_text_len,
        image_latent_dim,
        patch_size,
        latent_width,
        latent_height,
        pad_id,
        bos_id,
        eos_id,
        boi_id,
        eoi_id,
        bov_id,
        eov_id,
        img_pad_id,
        vid_pad_id,
        guidance_scale,
    ) = get_hyper_params(config, text_tokenizer, showo_token_ids)

    temperature = 1.0
    top_k = 1

    # -------------------------------------------------------------------------
    # Input images
    # -------------------------------------------------------------------------
    image_path_lower = config.mmu_image_path.lower()
    if image_path_lower.endswith((".jpg", ".jpeg", ".png")):
        file_list = [config.mmu_image_path]
    else:
        file_list = [
            os.path.join(config.mmu_image_path, fn)
            for fn in os.listdir(config.mmu_image_path)
            if fn.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

    if len(file_list) == 0:
        raise RuntimeError(f"No JPG/JPEG/PNG images found in: {config.mmu_image_path}")

    # Whole file = one prompt, including multiple lines.
    with open(config.prompt_file, "r", encoding="utf-8") as f:
        config.question = [f.read().strip()]

    sys_prompt_ids = text_tokenizer(
        "system\nYou are a helpful assistant.<|im_end|>",
        add_special_tokens=False,
    )["input_ids"]

    role_a = text_tokenizer(
        "\n<|im_start|>user\n",
        add_special_tokens=False,
    )["input_ids"]

    role_b = text_tokenizer(
        "\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )["input_ids"]

    # =========================================================================
    # STAGE 1: Load ONLY the VAE and precompute image latents.
    #
    # This prevents the VAE and the large Show-o2 checkpoint from occupying
    # GPU memory at the same time.
    # =========================================================================
    if config.model.vae_model.type == "wan21":
        from models import WanVAE

        vae_model = WanVAE(
            vae_pth=config.model.vae_model.pretrained_model_path,
            dtype=weight_type,
            device=device,
        )
    else:
        raise NotImplementedError

    cached_inputs = []

    print("Encoding image(s) with Wan VAE...", flush=True)

    with torch.inference_mode():
        for image_path in tqdm(file_list, desc="VAE encoding"):
            image_ori = Image.open(image_path).convert("RGB")

            image = image_transform(
                image_ori,
                resolution=config.dataset.preprocessing.resolution,
            ).to(device)
            image = image.unsqueeze(0)

            image_latents = (
                vae_model.sample(image.unsqueeze(2))
                .squeeze(2)
                .to(weight_type)
            )

            # Keep only the small latent on CPU while Show-o2 is being loaded.
            cached_inputs.append(
                {
                    "path": image_path,
                    "latents": image_latents.cpu(),
                }
            )

            del image_latents
            del image
            del image_ori

    # Completely remove VAE before loading Show-o2.
    del vae_model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(
            f"GPU allocated after deleting VAE: "
            f"{torch.cuda.memory_allocated() / 1024**3:.2f} GiB",
            flush=True,
        )
        print(
            f"GPU reserved after deleting VAE: "
            f"{torch.cuda.memory_reserved() / 1024**3:.2f} GiB",
            flush=True,
        )

    # =========================================================================
    # STAGE 2: Load Show-o2 only AFTER the VAE has been removed.
    # =========================================================================
    print("Loading Show-o2...", flush=True)

    if config.model.showo.load_from_showo:
        model = Showo2Qwen2_5.from_pretrained(
            config.model.showo.pretrained_model_path,
            torch_dtype=weight_type,
            low_cpu_mem_usage=True,
            use_safetensors=False,
        ).to(device)
    else:
        model = Showo2Qwen2_5(**config.model.showo).to(device)
        state_dict = load_state_dict(config.model_path)
        model.load_state_dict(state_dict)
        del state_dict
        gc.collect()

    model.eval()

    if torch.cuda.is_available():
        print(
            f"GPU allocated after loading Show-o2: "
            f"{torch.cuda.memory_allocated() / 1024**3:.2f} GiB",
            flush=True,
        )

    # =========================================================================
    # STAGE 3: MMU inference from cached VAE latents.
    # =========================================================================
    for step, cached in enumerate(tqdm(cached_inputs, desc="MMU inference")):
        image_path = cached["path"]

        image_latents = cached["latents"].to(
            device=device,
            dtype=weight_type,
        )

        response_text = ""

        with torch.inference_mode():

            image_embeds_und = model.image_embedder_und(image_latents)
            image_embeds_gen = model.image_embedder_gen(image_latents)

            image_embeds_und = (
                image_embeds_und
                + model.position_embedding(model.image_position_ids)
            )
            image_embeds_und = model.und_trans(
                image_embeds_und
            )["last_hidden_state"]

            image_embeds = model.fusion_proj(
                torch.cat(
                    [image_embeds_und, image_embeds_gen],
                    dim=-1,
                )
            )

            for question in config.question:
                input_ids = text_tokenizer(
                    question,
                    add_special_tokens=False,
                ).input_ids

                text_tokens_a = torch.tensor(
                    [showo_token_ids["bos_id"]] + sys_prompt_ids + role_a,
                    device=device,
                )[None, :]

                text_tokens_b = torch.tensor(
                    [showo_token_ids["boi_id"], showo_token_ids["eoi_id"]]
                    + input_ids
                    + role_b,
                    device=device,
                )[None, :]

                text_embeds_a = model.showo.model.embed_tokens(text_tokens_a)
                text_embeds_b = model.showo.model.embed_tokens(text_tokens_b)

                if config.model.showo.add_time_embeds:
                    time_embeds = model.time_embed(
                        torch.tensor([[1.0]], device=device),
                        text_embeds_a.dtype,
                    )

                    if hasattr(model, "time_embed_proj"):
                        time_embeds = model.time_embed_proj(time_embeds)

                    input_embeds = torch.cat(
                        [
                            text_embeds_a,
                            text_embeds_b[:, :1],
                            time_embeds,
                            image_embeds,
                            text_embeds_b[:, 1:],
                        ],
                        dim=1,
                    ).to(weight_type)

                    modality_positions = torch.tensor(
                        [text_tokens_a.shape[1] + 2, num_mmu_image_tokens],
                        device=device,
                    )[None, None, :]
                else:
                    input_embeds = torch.cat(
                        [
                            text_embeds_a,
                            text_embeds_b[:, :1],
                            image_embeds,
                            text_embeds_b[:, 1:],
                        ],
                        dim=1,
                    ).to(weight_type)

                    modality_positions = torch.tensor(
                        [text_tokens_a.shape[1] + 1, num_mmu_image_tokens],
                        device=device,
                    )[None, None, :]

                attention_mask = omni_attn_mask_naive(
                    B=input_embeds.size(0),
                    LEN=input_embeds.size(1),
                    modalities=modality_positions,
                    device=device,
                    inverted=True,
                ).to(input_embeds.dtype)

                output_tokens = model.mmu_generate(
                    input_embeds=input_embeds,
                    attention_mask=attention_mask,
                    temperature=temperature,
                    top_k=top_k,
                    max_new_tokens=300,
                    eos_token=text_tokenizer.eos_token_id,
                )

                output_tokens = torch.stack(output_tokens).squeeze()[None]

                text = text_tokenizer.batch_decode(
                    output_tokens,
                    skip_special_tokens=True,
                )

                print(f"\nImage: {image_path}", flush=True)
                print("Generated:", text[0], flush=True)

                response_text += (
                    f"User: {question}\n"
                    f"Answer: {text[0]}\n"
                )

                # Release prompt-specific tensors before the next prompt/image.
                del input_embeds
                del attention_mask
                del output_tokens
                del text_embeds_a
                del text_embeds_b
                del text_tokens_a
                del text_tokens_b

        # Log the original PIL image, without recreating a GPU image tensor.
        image_for_wandb = Image.open(image_path).convert("RGB")
        wandb.log(
            {
                "Multimodal understanding responses": [
                    wandb.Image(
                        image_for_wandb,
                        caption=response_text,
                    )
                ]
            },
            step=step,
        )
        image_for_wandb.close()

        del image_latents
        del image_embeds_und
        del image_embeds_gen
        del image_embeds

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    wandb.finish()