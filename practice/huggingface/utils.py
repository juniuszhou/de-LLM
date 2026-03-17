from huggingface_hub import login
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


def login_huggingface():
    token = read_env()
    login(new_session=True, token=token)
    print("Logged in to Hugging Face")


def read_env():
    with open("../../.env", "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.strip().split("=")
            print(key, value)
            if key == "HF_KEY":
                return value


# Paths relative to project root (parent of practice/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models"
DEFAULT_DATA_PATH = PROJECT_ROOT / "data"
DEFAULT_RESULTS_PATH = Path(__file__).resolve().parent / "results"


def resolve_model_path(model_name: str) -> Path:
    """Resolve model path, trying common locations."""
    path = PROJECT_ROOT / "models" / model_name
    print(path)
    if path.exists() and (path / "config.json").exists():
        return path
    raise FileNotFoundError(
        f"Model {model_name} not found"
    )  # Return as-is, let from_pretrained fail


def load_model_and_tokenizer(model_name):
    model_path = resolve_model_path(Path(model_name))
    model = AutoModelForCausalLM.from_pretrained(str(model_path))
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    return model, tokenizer


def load_local_dataset(data_set: str, config: str = "Mid", split: str = "train"):
    data_path = DEFAULT_DATA_PATH / data_set / config
    print("data_path is ", data_path)
    if data_path.exists():
        parquet_files = list(data_path.glob("*.parquet"))
        # all data files as train split
        data_files = {
            "train": [str(f) for f in parquet_files[:1]],
            "test": [str(f) for f in parquet_files[1:]],
        }
        # data_files is a dictionary of data files, the key is the split name, the value is a list of file paths
        dataset = load_dataset("parquet", data_files=data_files, split=split)
        return dataset
    else:
        raise FileNotFoundError(f"Dataset {data_set} not found")


def save_results(result_name: str):
    results_path = DEFAULT_RESULTS_PATH / result_name
    return results_path


if __name__ == "__main__":
    login_huggingface()
    load_model_and_tokenizer("SmolLM2-360M")
    load_local_dataset("smoltalk2")
    print(read_env())
