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

## examples

https://docs.tplr.ai/





## colab plugin usage
!nvidia-smi