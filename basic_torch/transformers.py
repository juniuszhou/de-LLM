import torch
import torch.nn as nn
import math

class TransformerModel(nn.Module):
    def __init__(self, ntoken=200, d_model=512, nhead=8, d_hid=2048, nlayers=2, dropout=0.2, pos_encoding_type="sinusoidal"):
        super().__init__()
        # Choose positional encoding type
        if pos_encoding_type == "sinusoidal":
            self.pos_encoder = PositionalEncoding(d_model, dropout)
        elif pos_encoding_type == "learnable":
            self.pos_encoder = LearnablePositionalEncoding(d_model, dropout=dropout)
        else:
            raise ValueError("pos_encoding_type must be 'sinusoidal' or 'learnable'")
            
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)

    def forward(self, src, src_mask=None):
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output

# Option 1: Sinusoidal Positional Encoding (Original Transformer paper)
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
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# Option 2: Learnable Positional Embedding (Alternative approach)
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Learnable positional embeddings
        self.pos_embedding = nn.Embedding(max_len, d_model)
    
    def forward(self, x):
        seq_len = x.size(0)
        positions = torch.arange(seq_len, device=x.device)
        pos_encodings = self.pos_embedding(positions).unsqueeze(1)
        x = x + pos_encodings
        return self.dropout(x)

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def main():
    # Example usage
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test both positional encoding types
    print("=== Testing Sinusoidal Positional Encoding ===")
    model_sin = TransformerModel(pos_encoding_type="sinusoidal").to(device)
    
    print("=== Testing Learnable Positional Encoding ===")
    model_learnable = TransformerModel(pos_encoding_type="learnable").to(device)
    
    # Example input (batch_size=1, sequence_length=10)
    src = torch.randint(0, 200, (10, 1)).to(device)
    print(f"Input tokens: {src.squeeze()}")
    src_mask = generate_square_subsequent_mask(src.size(0)).to(device)
    
    # Forward pass with sinusoidal encoding
    output_sin = model_sin(src, src_mask)
    print(f"Sinusoidal output shape: {output_sin.shape}")
    
    # Forward pass with learnable encoding
    output_learnable = model_learnable(src, src_mask)
    print(f"Learnable output shape: {output_learnable.shape}")
    
    # Compare parameter counts
    sin_params = sum(p.numel() for p in model_sin.parameters() if p.requires_grad)
    learnable_params = sum(p.numel() for p in model_learnable.parameters() if p.requires_grad)
    print(f"Sinusoidal model parameters: {sin_params:,}")
    print(f"Learnable model parameters: {learnable_params:,}")
    print(f"Difference: {learnable_params - sin_params:,} (learnable has more)")

if __name__ == "__main__":
    main()
