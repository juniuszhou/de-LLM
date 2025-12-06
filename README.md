# de-LLM

## venv local

python3 -m venv .venv && source .venv/bin/activate

## install dependency

curl -LsSf https://astral.sh/uv/install.sh | sh

### install according to pyproject.toml

uv venv # Create .venv directory

uv pip install ipykernel

uv sync

## add dependency

uv add torchtext
uv add transformers
uv add huggingface_hub
uv add langchain
uv add langchain-community
uv add langchain-openai

## examples

https://docs.tplr.ai/


## colab plugin usage
!nvidia-smi


## plan 12-1 to 12-7
day: langchain


weekend: complete gpt2  nanoGPT implementation follow the video from karpathy


12-8 to 12-14
## pytorch
