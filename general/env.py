import os

def read_env():
    with open(".env", "r") as f:
        for line in f:
            print(line)
            if line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.strip().split("=")
            if key  == "HF_KEY":
                print(value)

if __name__ == "__main__":
    read_env()