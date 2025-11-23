import torch
import torch.nn as nn
import torch.optim as optim
import math

# --- Configuration ---
# Small model configuration as requested
D_MODEL = 16
NHEAD = 2
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2
DIM_FEEDFORWARD = 64
DROPOUT = 0.1
BATCH_SIZE = 10
SEQ_LEN = 5
NUM_EPOCHS = 2000
LEARNING_RATE = 0.001
VOCAB_SIZE = 10  # Smaller vocab for easier learning

# Set seed for reproducibility
torch.manual_seed(42)


# --- Data Generation ---
# Task: Copy the input sequence.
# Input:  [A, B, C, D, E]
# Target: [A, B, C, D, E]
def generate_data(batch_size, seq_len, vocab_size):
    # Random sequences
    data = torch.randint(1, vocab_size, (batch_size, seq_len))
    return data, data # Source and Target are the same for copy task

# --- Model Definition ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [seq_len, batch_size, d_model]
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class SmallTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout):
        super().__init__()
        self.d_model = d_model
        
        # Embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # Output projection
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # src: [seq_len, batch_size]
        # tgt: [seq_len, batch_size]
        
        # Embed and add position info
        src = self.embedding(src) * math.sqrt(self.d_model)
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)
        
        # Transformer forward pass
        # Note: nn.Transformer takes [seq_len, batch_size, d_model] by default
        output = self.transformer(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        
        # Project to vocab size
        output = self.fc_out(output)
        return output

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

# --- Training Loop ---
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = SmallTransformer(
        VOCAB_SIZE, D_MODEL, NHEAD, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, DIM_FEEDFORWARD, DROPOUT
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.train()
    
    print("\nStarting training...")
    print("-" * 20)

    for epoch in range(NUM_EPOCHS):
        # Generate batch
        src_data, tgt_data = generate_data(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
        
        # Prepare inputs for Transformer (seq_len, batch_size)
        src = src_data.transpose(0, 1).to(device) # [SEQ_LEN, BATCH_SIZE]
        tgt = tgt_data.transpose(0, 1).to(device) # [SEQ_LEN, BATCH_SIZE]
        
        # Target input for decoder (shifted right) and Target output for loss
        # For simplicity in this demo, we'll just use the full sequence for training
        # In a real autoregressive task, decoder input is shifted.
        # Here we just want to show the mechanics.
        tgt_input = tgt[:-1, :]
        tgt_output = tgt[1:, :]

        src_mask = None
        tgt_mask = generate_square_subsequent_mask(tgt_input.size(0)).to(device)

        optimizer.zero_grad()
        
        # Forward
        output = model(src, tgt_input, src_mask, tgt_mask)
        # output: [seq_len-1, batch_size, vocab_size]
        
        # Loss
        loss = criterion(output.reshape(-1, VOCAB_SIZE), tgt_output.reshape(-1))
        
        # Backward
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 200 == 0:
            print(f"Training data: {src_data}")
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}")

    print("-" * 20)
    print("Training finished.")
    
    # --- Inference / Verification ---
    print("\nVerifying with a test sequence...")
    model.eval()
    with torch.no_grad():
        # Test sequence must be within vocab size (1 to VOCAB_SIZE-1)
        # VOCAB_SIZE is 4, so valid tokens are 1, 2, 3
        test_seq = generate_data(1, SEQ_LEN, VOCAB_SIZE)[0].transpose(0, 1).to(device)
        # test_seq = torch.tensor([[1, 2, 3, 1, 2]]).transpose(0, 1).to(device) # [SEQ_LEN, 1]
        # Start with a start token or just the first token? 
        # For this copy task, let's try to feed the source and see if it reconstructs.
        # In true inference we'd generate token by token.
        
        # Simple check: Feed the whole sequence (shifted) and see if it predicts the next tokens
        src = test_seq
        tgt_input = test_seq[:-1, :]
        tgt_mask = generate_square_subsequent_mask(tgt_input.size(0)).to(device)
        
        output = model(src, tgt_input, None, tgt_mask)
        predictions = output.argmax(dim=2)
        
        print(f"Input:      {test_seq.flatten().tolist()}")
        print(f"Target:     {test_seq[1:].flatten().tolist()}")
        print(f"Prediction: {predictions.flatten().tolist()}")

if __name__ == "__main__":
    train()
