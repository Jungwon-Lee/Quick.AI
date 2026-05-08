# SmallThinker-21BA3B-Instruct

This directory is a Quick.AI CLI template for
`Tiiny/SmallThinker-21BA3B-Instruct`.

Copy the tokenizer files from Hugging Face into this directory, then convert
the safetensor weights with the shared SmallThinker converter:

```bash
python3 ../SmallThinker-4BA0.6B-Instruct/download_and_convert.py \
  --repo_id Tiiny/SmallThinker-21BA3B-Instruct \
  --output_dir ./res/smallthinker/SmallThinker-21BA3B-Instruct \
  --mode all
```

The generated NNTrainer weight file must match `model_file_name` in
`nntr_config.json`.
