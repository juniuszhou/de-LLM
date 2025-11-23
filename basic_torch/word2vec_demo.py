import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, Counter
import random

class SimpleWord2Vec(nn.Module):
    """Simple Word2Vec model using Skip-gram approach"""
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        # Input word embeddings (center word)
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # Output word embeddings (context words)  
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Initialize embeddings with small random values
        self.center_embeddings.weight.data.uniform_(-0.1, 0.1)
        self.context_embeddings.weight.data.uniform_(-0.1, 0.1)
    
    def forward(self, center_word, context_words):
        # Get center word embedding
        center_embed = self.center_embeddings(center_word)  # [batch_size, embedding_dim]
        
        # Get context word embeddings
        context_embed = self.context_embeddings(context_words)  # [batch_size, num_context, embedding_dim]
        
        # Compute dot product scores
        scores = torch.bmm(context_embed, center_embed.unsqueeze(2)).squeeze(2)  # [batch_size, num_context]
        
        return scores
    
    def get_word_embedding(self, word_idx):
        """Get the final embedding for a word (average of center and context)"""
        with torch.no_grad():
            center_embed = self.center_embeddings.weight[word_idx]
            context_embed = self.context_embeddings.weight[word_idx]
            return (center_embed + context_embed) / 2

class Word2VecTrainer:
    def __init__(self, sentences, embedding_dim=50, window_size=2, learning_rate=0.01):
        self.sentences = sentences
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.learning_rate = learning_rate
        
        # Build vocabulary
        self.build_vocabulary()
        
        # Create model
        self.model = SimpleWord2Vec(len(self.vocab), embedding_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        # Generate training data
        self.training_data = self.generate_training_data()
        
    def build_vocabulary(self):
        """Build vocabulary from sentences"""
        word_counts = Counter()
        for sentence in self.sentences:
            for word in sentence.split():
                word_counts[word] += 1
        
        # Create word to index mapping
        self.vocab = list(word_counts.keys())
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        
        print(f"Vocabulary size: {len(self.vocab)}")
        print(f"Vocabulary: {self.vocab}")
        
    def generate_training_data(self):
        """Generate (center_word, context_word) pairs"""
        training_pairs = []
        
        for sentence in self.sentences:
            words = sentence.split()
            for i, center_word in enumerate(words):
                # Get context words within window
                start = max(0, i - self.window_size)
                end = min(len(words), i + self.window_size + 1)
                
                for j in range(start, end):
                    if i != j:  # Don't include the center word itself
                        context_word = words[j]
                        training_pairs.append((center_word, context_word))
        
        print(f"Generated {len(training_pairs)} training pairs")
        return training_pairs
    
    def train(self, epochs=100, print_every=20):
        """Train the word2vec model"""
        print(f"\nStarting training for {epochs} epochs...")
        print("=" * 60)
        
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            random.shuffle(self.training_data)
            
            for center_word, context_word in self.training_data:
                # Convert words to indices
                center_idx = torch.tensor([self.word_to_idx[center_word]])
                context_idx = self.word_to_idx[context_word]
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass: get scores for all possible context words
                all_context_words = torch.arange(len(self.vocab)).unsqueeze(0)  # All vocab as context
                scores = self.model(center_idx, all_context_words)
                
                # Compute loss (predict the correct context word)
                target = torch.tensor([context_idx])
                loss = self.criterion(scores, target)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(self.training_data)
            losses.append(avg_loss)
            
            if (epoch + 1) % print_every == 0:
                print(f"Epoch {epoch + 1:3d}/{epochs}: Average Loss = {avg_loss:.4f}")
        
        print("Training completed!")
        return losses
    
    def get_word_similarities(self, word, top_k=3):
        """Find most similar words to a given word"""
        if word not in self.word_to_idx:
            print(f"Word '{word}' not in vocabulary")
            return []
        
        word_idx = self.word_to_idx[word]
        word_embedding = self.model.get_word_embedding(word_idx)
        
        similarities = []
        for other_word, other_idx in self.word_to_idx.items():
            if other_word != word:
                other_embedding = self.model.get_word_embedding(other_idx)
                # Cosine similarity
                similarity = F.cosine_similarity(word_embedding, other_embedding, dim=0)
                similarities.append((other_word, similarity.item()))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def visualize_embeddings(self):
        """Visualize word embeddings in 2D using PCA"""
        try:
            from sklearn.decomposition import PCA
        except ImportError:
            print("sklearn not available. Cannot visualize embeddings.")
            return
        
        # Get all word embeddings
        embeddings = []
        words = []
        
        for word, idx in self.word_to_idx.items():
            embedding = self.model.get_word_embedding(idx).numpy()
            embeddings.append(embedding)
            words.append(word)
        
        # Reduce to 2D using PCA
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)
        
        for i, word in enumerate(words):
            plt.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=12)
        
        plt.title('Word Embeddings Visualization (PCA)')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

def demonstrate_word2vec_training():
    """Demonstrate the complete word2vec training process"""
    print("=" * 80)
    print("WORD2VEC TRAINING DEMONSTRATION")
    print("=" * 80)
    
    # Simple corpus with 10 words
    sentences = [
        "cat sat on mat",
        "dog ran in park", 
        "cat and dog are pets",
        "pets love their owners",
        "owners feed their pets",
        "cat loves fish food",
        "dog plays in park",
        "mat is soft and warm",
        "park has green grass",
        "fish swim in water"
    ]
    
    print("Training corpus:")
    for i, sentence in enumerate(sentences, 1):
        print(f"  {i}. {sentence}")
    
    # Create trainer
    trainer = Word2VecTrainer(
        sentences=sentences,
        embedding_dim=20,  # Small embedding dimension for demo
        window_size=2,     # Context window size
        learning_rate=0.01
    )
    
    print(f"\nTraining data examples:")
    for i, (center, context) in enumerate(trainer.training_data[:10]):
        print(f"  {i+1}. Center: '{center}' -> Context: '{context}'")
    if len(trainer.training_data) > 10:
        print(f"  ... and {len(trainer.training_data) - 10} more pairs")
    
    # Train the model
    losses = trainer.train(epochs=200, print_every=50)
    
    # Show results
    print("\n" + "=" * 60)
    print("TRAINING RESULTS")
    print("=" * 60)
    
    # Test word similarities
    test_words = ["cat", "dog", "park", "pets"]
    print("\nWord similarities:")
    for word in test_words:
        if word in trainer.word_to_idx:
            similarities = trainer.get_word_similarities(word, top_k=3)
            print(f"\nMost similar to '{word}':")
            for similar_word, score in similarities:
                print(f"  {similar_word}: {score:.4f}")
    
    # Show embeddings
    print(f"\nLearned embeddings (first 5 dimensions):")
    for word in trainer.vocab:
        idx = trainer.word_to_idx[word]
        embedding = trainer.model.get_word_embedding(idx)
        print(f"  {word:8s}: {embedding[:5].tolist()}")
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Visualize embeddings
    trainer.visualize_embeddings()
    
    return trainer

def show_embedding_evolution():
    """Show how embeddings evolve during training"""
    print("\n" + "=" * 60)
    print("EMBEDDING EVOLUTION DURING TRAINING")
    print("=" * 60)
    
    sentences = [
        "cat sat on mat",
        "dog ran in park", 
        "cat and dog are pets"
    ]
    
    trainer = Word2VecTrainer(sentences, embedding_dim=10, window_size=1, learning_rate=0.05)
    
    # Track embeddings at different training stages
    embedding_history = {word: [] for word in trainer.vocab}
    
    print("Tracking embedding evolution for words:", list(trainer.vocab))
    
    # Train and save embeddings at intervals
    training_stages = [0, 10, 50, 100, 200]
    
    for stage in training_stages:
        if stage == 0:
            # Initial embeddings
            pass
        else:
            # Train for some epochs
            prev_stage = training_stages[training_stages.index(stage) - 1]
            epochs_to_train = stage - prev_stage
            trainer.train(epochs=epochs_to_train, print_every=999)  # Don't print during training
        
        # Save current embeddings
        print(f"\nEmbeddings after {stage} epochs:")
        for word in trainer.vocab:
            idx = trainer.word_to_idx[word]
            embedding = trainer.model.get_word_embedding(idx)
            embedding_history[word].append(embedding.clone())
            print(f"  {word:6s}: {embedding[:3].tolist()} ...")
    
    # Show how specific words changed
    print(f"\nEvolution of 'cat' embedding:")
    for i, stage in enumerate(training_stages):
        embedding = embedding_history['cat'][i]
        print(f"  Epoch {stage:3d}: {embedding[:5].tolist()}")

def demonstrate_context_prediction():
    """Show how the model predicts context words"""
    print("\n" + "=" * 60)
    print("CONTEXT WORD PREDICTION")
    print("=" * 60)
    
    sentences = ["cat sat on mat", "dog ran in park", "cat and dog are pets"]
    trainer = Word2VecTrainer(sentences, embedding_dim=15, window_size=2)
    
    # Train briefly
    trainer.train(epochs=100, print_every=999)
    
    # Test prediction
    test_word = "cat"
    if test_word in trainer.word_to_idx:
        center_idx = torch.tensor([trainer.word_to_idx[test_word]])
        all_context_words = torch.arange(len(trainer.vocab)).unsqueeze(0)
        
        with torch.no_grad():
            scores = trainer.model(center_idx, all_context_words)
            probabilities = F.softmax(scores, dim=1)
        
        print(f"Context word probabilities for center word '{test_word}':")
        word_probs = []
        for i, word in enumerate(trainer.vocab):
            prob = probabilities[0, i].item()
            word_probs.append((word, prob))
        
        # Sort by probability
        word_probs.sort(key=lambda x: x[1], reverse=True)
        
        for word, prob in word_probs:
            print(f"  {word:8s}: {prob:.4f}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    # Run demonstrations
    trainer = demonstrate_word2vec_training()
    show_embedding_evolution()
    demonstrate_context_prediction()
