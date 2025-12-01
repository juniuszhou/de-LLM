from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import sys
import os
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.login_huggingface import login_huggingface
from sklearn.metrics.pairwise import cosine_similarity

def main():
    login_huggingface()
    model_id = "meta-llama/Llama-3.2-1B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    # all parameters number
    print(model.num_parameters())

    # for name, param in model.named_parameters():
    #     print(f"{name:80}  |  shape: {tuple(param.shape)}")

    all_params = model.state_dict()
    # print(all_params.keys())

    # weight matrix of the last layer
    weight = model.state_dict()["model.layers.15.mlp.up_proj.weight"].numpy()

    # plt.imshow(weight[:100, :100], cmap="RdBu")
    # plt.colorbar()
    # plt.show()

    input_embedding = model.state_dict()["model.embed_tokens.weight"].numpy()
    print(input_embedding.shape)

    token_id = 2
    token = tokenizer.decode(token_id)
    print(token)

    token_embedding = [input_embedding[token_id]]
    print(len(token_embedding))
    # print(token_embedding.shape, token_embedding)

    
    sims = cosine_similarity(token_embedding, input_embedding)[0]
    print(len(sims))
    print(sims)

if __name__ == "__main__":
    main()

