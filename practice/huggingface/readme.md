# hugging face

## download data

```shell
hf download HuggingFaceTB/smoltalk2 --repo-type dataset --local-dir ./data/smoltalk2
```

## download model

```shell
# SmolLM2-360M (smaller)
hf download HuggingFaceTB/SmolLM2-360M --local-dir ./SmolLM2-360M

# SmolLM3-3B-Base (for SFT)
hf download HuggingFaceTB/SmolLM3-3B-Base --local-dir ./model
```

## SFT (supervised fine-tuning)

Run SFT with SmolLM3-3B-Base and Smoltalk2:

```shell
python practice/huggingface/sft.py --model-path ./model --data-path ./data/smoltalk2
```

Options: `--max-samples`, `--num-epochs`, `--batch-size`, `--lr`, `--no-4bit`, `--split`, etc.
