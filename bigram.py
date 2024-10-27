import torch
import torch.nn as nn
from torch.nn import functional as F

class Head(nn.Module):
    """One head of self-attention."""

    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()
        # Linear layers to project input embeddings to key, query, and value vectors
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        
        # Create a lower triangular matrix for masking future tokens
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape  # B: batch size, T: sequence length, C: embedding dimension
        
        # Compute key, query, and value matrices
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)

        # Compute attention scores (affinities) using scaled dot-product attention
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        
        # Apply the triangular mask to prevent attending to future tokens
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        
        # Convert scores to probabilities
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        
        # Apply dropout to attention weights
        wei = self.dropout(wei)
        
        # Compute value representations
        v = self.value(x)  # (B, T, head_size)
        
        # Aggregate values based on attention weights
        out = wei @ v  # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel."""

    def __init__(self, num_heads, head_size, n_embd, dropout, block_size):
        super().__init__()
        # Create multiple attention heads
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])
        # Final linear layer to combine heads
        self.proj = nn.Linear(n_embd, n_embd)
        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Concatenate outputs from all attention heads
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, num_heads * head_size)
        
        # Project back to the original embedding size and apply dropout
        out = self.dropout(self.proj(out))  # (B, T, n_embd)
        return out


class FeedFoward(nn.Module):
    """A simple linear layer followed by a non-linearity."""

    def __init__(self, n_embd, dropout):
        super().__init__()
        # Feedforward network with two linear layers and ReLU activation
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # Increase dimensionality
            nn.ReLU(),                      # Non-linearity
            nn.Linear(4 * n_embd, n_embd),  # Reduce back to original size
            nn.Dropout(dropout),            # Regularization
        )

    def forward(self, x):
        return self.net(x)  # Forward pass through the feedforward network


class Block(nn.Module):
    """Transformer block: communication followed by computation."""

    def __init__(self, n_embd, n_head, dropout, block_size):
        super().__init__()
        head_size = n_embd // n_head  # Size of each attention head
        # Initialize multi-head attention and feedforward layers
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, dropout, block_size)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)  # Layer normalization
        self.ln2 = nn.LayerNorm(n_embd)  # Layer normalization

    def forward(self, x):
        # Apply self-attention and add residual connection
        x = x + self.sa(self.ln1(x))  # (B, T, n_embd)
        # Apply feedforward network and add residual connection
        x = x + self.ffwd(self.ln2(x))  # (B, T, n_embd)
        return x


# Super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self, device, config, vocab_size):
        super().__init__()
        self.block_size = config["block_size"]  # Max sequence length
        self.device = device  # Device to run on (CPU or GPU)

        # Embedding tables for tokens and positions
        self.token_embedding_table = nn.Embedding(vocab_size, config["n_embd"])  # (vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(config["block_size"], config["n_embd"])  # (block_size, n_embd)
        
        # Stack of transformer blocks
        self.blocks = nn.Sequential(*[Block(config["n_embd"], config["n_head"], config["dropout"], config["block_size"]) for _ in range(config["n_layer"])])
        
        self.ln_f = nn.LayerNorm(config["n_embd"])  # Final layer normalization
        self.lm_head = nn.Linear(config["n_embd"], vocab_size)  # Output layer to predict next token

    def forward(self, idx, targets=None):
        B, T = idx.shape  # B: batch size, T: sequence length

        # Token and position embeddings
        tok_emb = self.token_embedding_table(idx)  # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))  # (T, n_embd)
        
        # Combine token and position embeddings
        x = tok_emb + pos_emb  # (B, T, n_embd)
        
        # Pass through transformer blocks
        x = self.blocks(x)  # (B, T, n_embd)
        
        # Final layer normalization
        x = self.ln_f(x)  # (B, T, n_embd)
        
        # Compute logits for the next token
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Calculate loss if targets are provided
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape  # C: vocab_size
            logits = logits.view(B * T, C)  # Reshape for cross-entropy
            targets = targets.view(B * T)  # Reshape targets
            loss = F.cross_entropy(logits, targets)  # Compute loss

        return logits, loss  # Return logits and loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]  # Keep only the last block_size tokens
            
            # Get the predictions for the current context
            logits, loss = self(idx_cond)  # (B, T, vocab_size)
            
            # Focus only on the last time step's logits
            logits = logits[:, -1, :]  # (B, vocab_size)
            
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, vocab_size)
            
            # Sample from the distribution to get the next token index
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            
            # Append the sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        
        return idx  # Return the sequence including newly generated tokens
