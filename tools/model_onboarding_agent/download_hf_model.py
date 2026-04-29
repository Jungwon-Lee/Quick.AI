#!/usr/bin/env python3
"""Download and save a Hugging Face model with transformers."""

from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download HF model and tokenizer")
    parser.add_argument("--model", required=True, help="HF model id or URL")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--auto-class",
        choices=["causal-lm", "auto-model"],
        default="causal-lm",
        help="Use AutoModelForCausalLM or AutoModel",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--token", help="HF token for gated/private models")
    return parser.parse_args()


def normalize_model_name(model: str) -> str:
    prefix = "https://huggingface.co/"
    if model.startswith(prefix):
        return model[len(prefix) :].strip("/")
    return model


def main() -> None:
    args = parse_args()
    model_name = normalize_model_name(args.model)
    model_cls = AutoModelForCausalLM if args.auto_class == "causal-lm" else AutoModel

    model = model_cls.from_pretrained(
        model_name,
        revision=args.revision,
        token=args.token,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=args.revision,
        token=args.token,
        trust_remote_code=args.trust_remote_code,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved {model_name}@{args.revision} to {args.output_dir}")


if __name__ == "__main__":
    main()
