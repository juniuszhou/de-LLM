"""
Tokenization Examples: PyTorch and Hugging Face

This file demonstrates how to use tokenization in both PyTorch and Hugging Face.
"""

import torch
import re
from collections import Counter

# Hugging Face tokenization
from transformers import AutoTokenizer


def simple_tokenizer(text, lowercase=True):
    """
    Simple tokenizer that splits text on whitespace and punctuation.
    This is a basic PyTorch-compatible tokenizer without external dependencies.
    """
    if lowercase:
        text = text.lower()
    # Split on whitespace and punctuation, keeping alphanumeric sequences
    tokens = re.findall(r'\b\w+\b', text)
    return tokens


def pytorch_tokenization_example():
    """
    Example of tokenization using a simple PyTorch-compatible approach.
    Note: PyTorch doesn't have a built-in tokenizer, so we use a simple
    regex-based tokenizer or can use torchtext (which has compatibility issues).
    """
    print("=" * 60)
    print("PyTorch Tokenization Example")
    print("=" * 60)
    
    # Example text
    text = "Hello, how are you? I'm learning PyTorch tokenization!"
    
    # Tokenize the text using simple tokenizer
    tokens = simple_tokenizer(text)
    
    print(f"Original text: {text}")
    print(f"Tokens: {tokens}")
    print(f"Number of tokens: {len(tokens)}")
    print()
    
    # Create vocabulary and convert tokens to indices (PyTorch style)
    vocab = {word: idx for idx, word in enumerate(set(tokens))}
    token_indices = [vocab[token] for token in tokens]
    token_tensor = torch.tensor(token_indices)
    
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Token indices: {token_indices}")
    print(f"Token tensor: {token_tensor}")
    print()


def pytorch_tokenization_with_vocab():
    """
    Example of creating a vocabulary and tokenizing with PyTorch tensors.
    """
    print("=" * 60)
    print("PyTorch Tokenization with Vocabulary")
    print("=" * 60)
    
    # Example sentences
    sentences = [
        "Hello, how are you?",
        "I'm learning PyTorch tokenization.",
        "This is a great example!"
    ]
    
    # Tokenize all sentences
    all_tokens = []
    for sentence in sentences:
        tokens = simple_tokenizer(sentence)
        all_tokens.append(tokens)
    
    # Build vocabulary
    word_counter = Counter()
    for tokens in all_tokens:
        word_counter.update(tokens)
    
    # Create vocabulary with special tokens
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, count in word_counter.items():
        vocab[word] = len(vocab)
    
    # Convert tokens to indices
    tokenized_sentences = []
    for tokens in all_tokens:
        indices = [vocab.get(token, vocab['<UNK>']) for token in tokens]
        tokenized_sentences.append(indices)
    
    # Pad sequences to same length
    max_len = max(len(seq) for seq in tokenized_sentences)
    padded_sequences = []
    for seq in tokenized_sentences:
        padded = seq + [vocab['<PAD>']] * (max_len - len(seq))
        padded_sequences.append(padded)
    
    # Convert to PyTorch tensor
    token_tensor = torch.tensor(padded_sequences)
    
    print("Sentences:")
    for sentence in sentences:
        print(f"  {sentence}")
    print()
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Token tensor shape: {token_tensor.shape}")
    print(f"Token tensor:\n{token_tensor}")
    print()


def huggingface_tokenization_example():
    """
    Example of tokenization using Hugging Face transformers library.
    """
    print("=" * 60)
    print("Hugging Face Tokenization Example")
    print("=" * 60)
    
    # Load a pre-trained tokenizer (using BERT as an example)
    # You can use any model name from Hugging Face Hub
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Example text
    text = "Hello, how are you? I'm learning Hugging Face tokenization!"
    
    # Basic tokenization
    tokens = tokenizer.tokenize(text)
    print(f"Original text: {text}")
    print(f"Tokens: {tokens}")
    print(f"Number of tokens: {len(tokens)}")
    print()
    
    # Encode text to token IDs
    token_ids = tokenizer.encode(text)
    print(f"Token IDs: {token_ids}")
    print(f"Number of token IDs: {len(token_ids)}")
    print()
    
    # Encode with return_tensors (for model input)
    encoded = tokenizer(text, return_tensors="pt")
    print(f"Encoded (with tensors):")
    print(f"  input_ids: {encoded['input_ids']}")
    print(f"  attention_mask: {encoded['attention_mask']}")
    print()
    
    # Decode token IDs back to text
    decoded_text = tokenizer.decode(token_ids)
    print(f"Decoded text: {decoded_text}")
    print()


def huggingface_tokenization_advanced():
    """
    Advanced Hugging Face tokenization examples.
    """
    print("=" * 60)
    print("Advanced Hugging Face Tokenization")
    print("=" * 60)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Multiple sentences
    texts = [
        "Hello, how are you?",
        "I'm learning tokenization.",
        "This is a great example!"
    ]
    
    # Batch tokenization
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    
    print("Batch tokenization:")
    print(f"  input_ids shape: {encoded['input_ids'].shape}")
    print(f"  attention_mask shape: {encoded['attention_mask'].shape}")
    print()
    
    # Add special tokens
    encoded_with_special = tokenizer(
        texts[0],
        add_special_tokens=True,
        return_tensors="pt"
    )
    print(f"With special tokens: {encoded_with_special['input_ids']}")
    print()


def comparison_example():
    """
    Compare PyTorch and Hugging Face tokenization on the same text.
    """
    print("=" * 60)
    print("Comparison: PyTorch vs Hugging Face")
    print("=" * 60)
    
    text = "Hello, how are you? I'm learning tokenization!"
    
    # PyTorch tokenization (simple tokenizer)
    pytorch_tokens = simple_tokenizer(text)
    
    # Hugging Face tokenization
    hf_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    hf_tokens = hf_tokenizer.tokenize(text)
    
    print(f"Text: {text}")
    print()
    print(f"PyTorch tokens ({len(pytorch_tokens)}):")
    print(f"  {pytorch_tokens}")
    print()
    print(f"Hugging Face tokens ({len(hf_tokens)}):")
    print(f"  {hf_tokens}")
    print()


if __name__ == "__main__":
    # Run examples
    pytorch_tokenization_example()
    pytorch_tokenization_with_vocab()
    huggingface_tokenization_example()
    huggingface_tokenization_advanced()
    comparison_example()

