import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class MiniWord2Vec(nn.Module):
    """Minimal Word2Vec implementation for demonstration"""
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        # The embedding layer - this is where word vectors are learned
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # Output layer to predict context words
        self.output = nn.Linear(embedding_dim, vocab_size)
        
    def forward(self, center_word):
        # Get embedding for center word
        embedded = self.embeddings(center_word)  # [batch_size, embedding_dim]
        # Predict context words
        output = self.output(embedded)  # [batch_size, vocab_size]
        return output

def create_training_data():
    """Create simple training data with 10 words"""
    # Our mini vocabulary
    sentences = [
        "cat sits on mat",
        "dog runs fast", 
        "cat loves fish",
        "dog plays ball",
        "fish swims water",
        "ball bounces high",
        "mat is soft",
        "water is blue",
        "fast dog runs",
        "soft cat sits"
    ]
    
    # Build vocabulary
    words = set()
    for sentence in sentences:
        words.update(sentence.split())
    
    vocab = sorted(list(words))  # ['ball', 'blue', 'bounces', 'cat', 'dog', ...]
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    
    print(f"Vocabulary ({len(vocab)} words): {vocab}")
    
    # Create training pairs (center_word, context_word)
    training_pairs = []
    window_size = 1  # Look at 1 word on each side
    
    for sentence in sentences:
        words_in_sentence = sentence.split()
        for i, center_word in enumerate(words_in_sentence):
            # Get context words
            for j in range(max(0, i - window_size), min(len(words_in_sentence), i + window_size + 1)):
                if i != j:  # Don't include center word as its own context
                    context_word = words_in_sentence[j]
                    training_pairs.append((center_word, context_word))
    
    print(f"Training pairs ({len(training_pairs)} examples):")
    for i, (center, context) in enumerate(training_pairs[:15]):
        print(f"  {i+1:2d}. '{center}' -> '{context}'")
    if len(training_pairs) > 15:
        print(f"  ... and {len(training_pairs) - 15} more")
    
    return vocab, word_to_idx, training_pairs

def train_word2vec():
    """Train the word2vec model step by step"""
    print("=" * 60)
    print("TRAINING WORD2VEC FROM SCRATCH")
    print("=" * 60)
    
    # Create data
    vocab, word_to_idx, training_pairs = create_training_data()
    vocab_size = len(vocab)
    embedding_dim = 10  # Small embedding size for demo
    
    # Create model
    model = MiniWord2Vec(vocab_size, embedding_dim)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\nModel architecture:")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Embedding dimension: {embedding_dim}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Show initial embeddings
    print(f"\nInitial embeddings for first 3 words:")
    for i in range(3):
        word = vocab[i]
        embedding = model.embeddings.weight[i].detach()
        print(f"  {word:8s}: {embedding.tolist()}")
    
    # Training loop
    epochs = 50
    print(f"\nTraining for {epochs} epochs...")
    
    for epoch in range(epochs):
        total_loss = 0
        
        # Shuffle training data
        import random
        random.shuffle(training_pairs)
        
        for center_word, context_word in training_pairs:
            # Convert words to indices
            center_idx = torch.tensor([word_to_idx[center_word]])
            context_idx = torch.tensor([word_to_idx[context_word]])
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(center_idx)
            loss = criterion(outputs, context_idx)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(training_pairs)
            print(f"  Epoch {epoch + 1:2d}/{epochs}: Loss = {avg_loss:.4f}")
    
    print("Training completed!")
    
    # Show learned embeddings
    print(f"\nLearned embeddings:")
    for i, word in enumerate(vocab):
        embedding = model.embeddings.weight[i].detach()
        print(f"  {word:8s}: {embedding.tolist()}")
    
    # Test word similarities
    print(f"\nWord similarities (cosine similarity):")
    test_words = ["cat", "dog", "fish"]
    
    for word in test_words:
        if word in word_to_idx:
            word_idx = word_to_idx[word]
            word_embedding = model.embeddings.weight[word_idx]
            
            similarities = []
            for other_word, other_idx in word_to_idx.items():
                if other_word != word:
                    other_embedding = model.embeddings.weight[other_idx]
                    similarity = F.cosine_similarity(word_embedding, other_embedding, dim=0)
                    similarities.append((other_word, similarity.item()))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            print(f"\n  Most similar to '{word}':")
            for similar_word, score in similarities[:3]:
                print(f"    {similar_word:8s}: {score:.4f}")
    
    return model, vocab, word_to_idx

def demonstrate_prediction():
    """Show how the trained model makes predictions"""
    print("\n" + "=" * 60)
    print("PREDICTION DEMONSTRATION")
    print("=" * 60)
    
    model, vocab, word_to_idx = train_word2vec()
    
    # Test prediction for a specific word
    test_word = "cat"
    if test_word in word_to_idx:
        print(f"\nPredicting context words for '{test_word}':")
        
        word_idx = torch.tensor([word_to_idx[test_word]])
        
        with torch.no_grad():
            outputs = model(word_idx)
            probabilities = F.softmax(outputs, dim=1)
        
        # Get top predictions
        top_probs, top_indices = torch.topk(probabilities, k=5, dim=1)
        
        print("Top 5 predicted context words:")
        for i in range(5):
            pred_idx = top_indices[0, i].item()
            pred_word = vocab[pred_idx]
            pred_prob = top_probs[0, i].item()
            print(f"  {i+1}. {pred_word:8s}: {pred_prob:.4f}")

def show_training_process():
    """Show detailed training process step by step"""
    print("\n" + "=" * 60)
    print("DETAILED TRAINING PROCESS")
    print("=" * 60)
    
    # Simple example
    vocab = ["cat", "dog", "sits", "runs"]
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    
    model = MiniWord2Vec(len(vocab), 5)  # 5-dim embeddings
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()
    
    # One training example: "cat" -> "sits"
    center_word = "cat"
    context_word = "sits"
    
    center_idx = torch.tensor([word_to_idx[center_word]])
    context_idx = torch.tensor([word_to_idx[context_word]])
    
    print(f"Training example: '{center_word}' -> '{context_word}'")
    print(f"Center word index: {center_idx.item()}")
    print(f"Context word index: {context_idx.item()}")
    
    # Show one training step in detail
    print(f"\nBefore training step:")
    print(f"  Embedding for '{center_word}': {model.embeddings.weight[center_idx].detach().tolist()}")
    
    # Forward pass
    optimizer.zero_grad()
    outputs = model(center_idx)
    loss = criterion(outputs, context_idx)
    
    print(f"  Model output scores: {outputs.detach().tolist()}")
    print(f"  Loss: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    print(f"  Gradients computed!")
    
    # Update parameters
    optimizer.step()
    
    print(f"\nAfter training step:")
    print(f"  Embedding for '{center_word}': {model.embeddings.weight[center_idx].detach().tolist()}")
    print(f"  Embedding changed slightly through gradient descent!")

if __name__ == "__main__":
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    # Run demonstrations
    show_training_process()
    demonstrate_prediction()


