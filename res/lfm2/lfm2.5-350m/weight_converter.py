## @file weight_converter.py
## @brief weight conversion script for LFM2/LFM2.5 models

import argparse
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def save_lfm2_for_nntrainer(params, config, dtype, file):
    def save_weight(weight, out_dtype=dtype):
        np.array(weight.detach().cpu(), dtype=out_dtype).tofile(file)

    def save_linear(key):
        save_weight(params[key].permute(1, 0))

    save_weight(params["model.embed_tokens.weight"])

    for layer_idx, layer_type in enumerate(config.layer_types):
        prefix = f"model.layers.{layer_idx}."

        save_weight(params[f"{prefix}operator_norm.weight"])

        if layer_type == "full_attention":
            attn = prefix + "self_attn."
            save_linear(attn + "q_proj.weight")
            save_weight(params[attn + "q_layernorm.weight"])
            save_linear(attn + "k_proj.weight")
            save_weight(params[attn + "k_layernorm.weight"])
            save_linear(attn + "v_proj.weight")
            save_linear(attn + "out_proj.weight")
        else:
            conv = prefix + "conv."
            save_linear(conv + "in_proj.weight")
            save_weight(params[conv + "conv.weight"].squeeze(1), "float32")
            save_linear(conv + "out_proj.weight")

        save_weight(params[f"{prefix}ffn_norm.weight"])
        save_linear(prefix + "feed_forward.w1.weight")
        save_linear(prefix + "feed_forward.w3.weight")
        save_linear(prefix + "feed_forward.w2.weight")

    save_weight(params["model.embedding_norm.weight"])
    if "lm_head.weight" in params:
        save_weight(params["lm_head.weight"].permute(1, 0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./LFM2.5-350M")
    parser.add_argument("--output_name", type=str,
                        default="./nntr_lfm2.5_350m_fp32.bin")
    parser.add_argument("--data_type", type=str, default="float32")
    args = parser.parse_args()

    AutoTokenizer.from_pretrained(args.model_path)
    config = AutoConfig.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.float32
    )
    model.eval()

    with open(args.output_name, "wb") as f_model:
        save_lfm2_for_nntrainer(
            model.state_dict(), config, args.data_type, f_model
        )
