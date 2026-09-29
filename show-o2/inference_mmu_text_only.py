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

import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
from PIL import Image
import wandb
import torch
from tqdm import tqdm
from accelerate.logging import get_logger
from models import Showo2Qwen2_5, omni_attn_mask, omni_attn_mask_naive
from models.misc import get_text_tokenizer, prepare_gen_input
from utils import get_config, flatten_omega_conf, denorm, get_hyper_params, path_to_llm_name, load_state_dict, set_seed
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from datasets.utils import image_transform, resize_and_pad_image, to_tensor_and_normalize

# set_seed(10)

logger = get_logger(__name__, log_level="INFO")

if __name__ == '__main__':

    config = get_config()

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
    weight_type = torch.float16

    # VQ model for processing image into discrete tokens
    if config.model.vae_model.type == 'wan21':
        from models import WanVAE
        vae_model = WanVAE(vae_pth=config.model.vae_model.pretrained_model_path, dtype=weight_type, device=device)
    else:
        raise NotImplementedError

    # Initialize Show-o model
    text_tokenizer, showo_token_ids = get_text_tokenizer(config.model.showo.llm_model_path,
                                                         add_showo_tokens=True,
                                                         return_showo_token_ids=True,
                                                         llm_name=path_to_llm_name[config.model.showo.llm_model_path])
    config.model.showo.llm_vocab_size = len(text_tokenizer)

    if config.model.showo.load_from_showo:
        model = Showo2Qwen2_5.from_pretrained(config.model.showo.pretrained_model_path, torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,  use_safetensors=False).to(device)
    else:
        model = Showo2Qwen2_5(**config.model.showo).to(device)
        state_dict = load_state_dict(config.model_path)
        model.load_state_dict(state_dict)

    model.to(weight_type)
    model.eval()

    # for time embedding
    if config.model.showo.add_time_embeds:
        # we prepend the time embedding to vision tokens
        config.dataset.preprocessing.num_t2i_image_tokens += 1
        config.dataset.preprocessing.num_mmu_image_tokens += 1
        config.dataset.preprocessing.num_video_tokens += 1

    num_t2i_image_tokens, num_mmu_image_tokens, num_video_tokens, max_seq_len, max_text_len, image_latent_dim, patch_size, latent_width, \
    latent_height, pad_id, bos_id, eos_id, boi_id, eoi_id, bov_id, eov_id, img_pad_id, vid_pad_id, guidance_scale \
        = get_hyper_params(config, text_tokenizer, showo_token_ids)

    temperature = 1.0  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    top_k = 1  # retain only the top_k most likely tokens, clamp others to have 0 probability

    if not (config.mmu_image_path.endswith('.jpg') or config.mmu_image_path.endswith('.png')):
        file_list = [os.path.join(config.mmu_image_path, fn) for fn in os.listdir(config.mmu_image_path)]
    else:
        file_list = [config.mmu_image_path]

    #config.question = config.question.split(' *** ')
    with open(config.prompt_file, "r", encoding="utf-8") as f:
        config.question = [f.read().strip()]

    sys_prompt_ids = text_tokenizer("system\nYou are a helpful assistant.<|im_end|>",
                                    add_special_tokens=False)['input_ids']
    role_a = text_tokenizer("\n<|im_start|>user\n", add_special_tokens=False)['input_ids']
    role_b = text_tokenizer("\n<|im_start|>assistant\n", add_special_tokens=False)['input_ids']

    # =========================================================
    # TEXT-ONLY TEST OF SHOW-O2's INTERNAL QWEN
    # =========================================================

    scene_description = """
    The image contains:
    - a woman taking a selfie
    - a bed
    - a nightstand
    - a lamp
    - a framed coastal painting
    - an indoor private-room setting   
    """

    with open(config.prompt_file, "r", encoding="utf-8") as f:
        investigation_prompt = f.read().strip()

    question = scene_description + "\n\n" + investigation_prompt

    input_ids = (
            [showo_token_ids["bos_id"]]
            + sys_prompt_ids
            + role_a
            + text_tokenizer(
        question,
        add_special_tokens=False
    ).input_ids
            + role_b
    )

    generated_text = model.lm_generate(
        input_ids=input_ids,
        tokenizer=text_tokenizer,
        max_new_tokens=300,
        boi_token=showo_token_ids["boi_id"],
        temperature=1.0,
        top_k=1,
        device=device,
    )

    print("\nTEXT-ONLY SHOW-O2 QWEN OUTPUT:")
    print(generated_text)

    exit()


