from transformers import GPT2Tokenizer, GPT2Model, pipeline
import json


def main():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2")
    pipe = pipeline("text-generation", "gpt2")

    text = "who is worst president of USA in history."
    outputs = pipe(text, max_length=100, truncation=True, output_hidden_states=True)
    print(outputs)

    hidden_states = outputs[0].hidden_states

    for idx, h in enumerate(hidden_states):
        print(idx, h.shape)

    fixed_text = outputs[0]["generated_text"]

    print("outputs is: ", fixed_text.removeprefix(text))

    # parameters = model.named_parameters()
    # for name, param in parameters:
    #     print(name, param.shape)


if __name__ == "__main__":
    main()
