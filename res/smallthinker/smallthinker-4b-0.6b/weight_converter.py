## @file weight_converter.py
## @brief weight conversion script for SmallThinker 4B-A0.6B

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from types import SimpleNamespace
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM


class SafetensorParams:
    def __init__(self, model_path):
        model_path = Path(model_path)
        with (model_path / "model.safetensors.index.json").open() as f:
            index = json.load(f)
        self.model_path = model_path
        self.weight_map = index["weight_map"]

    def __getitem__(self, weight_name):
        shard_name = self.weight_map[weight_name]
        with safe_open(self.model_path / shard_name, framework="pt",
                       device="cpu") as f:
            return f.get_tensor(weight_name)


def load_config(model_path):
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if not hasattr(cfg, "tie_word_embeddings"):
        cfg.tie_word_embeddings = True
    return cfg


def load_params(model_path):
    index_path = Path(model_path) / "model.safetensors.index.json"
    if index_path.exists():
        return SafetensorParams(model_path)

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32, trust_remote_code=True)
    model.eval()
    return model.state_dict()


def save_smallthinker_for_nntrainer(params, config, dtype, file):
    """Convert and save SmallThinker weights in NNTrainer layer order."""

    def save_weight(weight_name, is_transpose=False):
        print(weight_name, params[weight_name].shape)
        weight = params[weight_name]
        if is_transpose:
            weight = weight.permute(1, 0)
        weight.detach().to(torch.float32).cpu().numpy().astype(dtype).tofile(file)

    def save_projection(layer_name, proj_name):
        save_weight(f"{layer_name}{proj_name}.weight", True)

    def save_attention(layer_name):
        save_weight(f"{layer_name}input_layernorm.weight")
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            save_projection(layer_name, f"self_attn.{proj}")

    def save_moe(layer_name):
        save_weight(f"{layer_name}block_sparse_moe.primary_router.weight", True)
        for expert_id in range(config.moe_num_primary_experts):
            expert = f"{layer_name}block_sparse_moe.experts.{expert_id}."
            for proj in ["up", "gate", "down"]:
                save_projection(expert, proj)

    save_weight("model.embed_tokens.weight")

    for layer_idx in range(config.num_hidden_layers):
        layer_prefix = f"model.layers.{layer_idx}."
        save_attention(layer_prefix)
        save_weight(f"{layer_prefix}post_attention_layernorm.weight")
        save_moe(layer_prefix)

    save_weight("model.norm.weight")
    if not config.tie_word_embeddings:
        save_weight("lm_head.weight", True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        default="./SmallThinker-4BA0.6B-Instruct")
    parser.add_argument("--output_name", type=str,
                        default="./nntr_smallthinker_4ba0.6b_fp32.bin")
    parser.add_argument("--data_type", type=str, default="float32")
    args = parser.parse_args()

    config = load_config(args.model_path)
    params = load_params(args.model_path)

    with open(args.output_name, "wb") as f_model:
        save_smallthinker_for_nntrainer(
            params, config, args.data_type, f_model)
