import torch
import torch.nn as nn
from torch.nn import functional as F
from datetime import datetime
import json

from bigram import BigramLanguageModel  # Import the BigramLanguageModel class

# Load configuration from JSON file
config = {}
with open("config.json", "r") as jsonfile:
    config = json.load(jsonfile)  # Read configuration settings
    print("Config: ", config)

# Determine the device to use (GPU or CPU)
device = 'cuda:0'  # Default to CUDA device 0
if config["allow_gpu"]:  # Check if GPU usage is allowed
    if torch.cuda.is_available():  # If CUDA is available, use it
        device = 'cuda' 
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():  # Check for Apple Silicon
        device = 'mps' 
print("Using device: " + device)  # Output the chosen device

# Start timer for training duration
start_time = datetime.now()
torch.manual_seed(1337)  # Set random seed for reproducibility

# Load training data
with open(config["data_path"], 'r', encoding='utf-8') as f:
    text = f.read()  # Read the entire text data

# Identify all unique characters in the text
chars = sorted(list(set(text)))  # Create a sorted list of unique characters
vocab_size = len(chars)  # Determine the size of the vocabulary
# Create mappings from characters to integers (stoi) and integers to characters (itos)
stoi = { ch: i for i, ch in enumerate(chars) }
itos = { i: ch for i, ch in enumerate(chars) }
# Define encoding and decoding functions
encode = lambda s: [stoi[c] for c in s]  # Encoder: converts string to list of integers
decode = lambda l: ''.join([itos[i] for i in l])  # Decoder: converts list of integers to string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)  # Encode text as tensor of integers
n = int(0.9 * len(data))  # 90% of the data for training
train_data = data[:n]  # Training data
val_data = data[n:]  # Validation data

# Function to create batches of data
def get_batch(split):
    # Generate a small batch of data for inputs x and targets y
    data = train_data if split == 'train' else val_data  # Choose the dataset
    ix = torch.randint(len(data) - config["block_size"], (config["batch_size"],))  # Random starting indices
    x = torch.stack([data[i:i + config["block_size"]] for i in ix])  # Input sequences
    y = torch.stack([data[i + 1:i + config["block_size"] + 1] for i in ix])  # Target sequences (next token)
    x, y = x.to(device), y.to(device)  # Move tensors to the selected device
    return x, y  # Return the input and target tensors

@torch.no_grad()  # Disable gradient calculation for efficiency
def estimate_loss():
    out = {}  # Dictionary to store loss estimates
    model.eval()  # Set model to evaluation mode
    for split in ['train', 'val']:  # Iterate over training and validation datasets
        losses = torch.zeros(config["eval_iters"])  # Initialize loss tensor
        for k in range(config["eval_iters"]):
            X, Y = get_batch(split)  # Get a batch of data
            logits, loss = model(X, Y)  # Forward pass through the model
            losses[k] = loss.item()  # Store the loss value
        out[split] = losses.mean()  # Average the losses for the split
    model.train()  # Set model back to training mode
    return out  # Return the loss estimates

# Initialize the Bigram Language Model
model = BigramLanguageModel(device=device, config=config, vocab_size=vocab_size)
m = model.to(device)  # Move model to the selected device
# Print the number of parameters in the model
print(sum(p.numel() for p in m.parameters()) / 1e6, 'M parameters')

# Create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])

# Main training loop
for iter in range(config["max_iters"]):

    # Evaluate the loss on training and validation sets at specified intervals
    if iter % config["eval_interval"] == 0 or iter == config["max_iters"] - 1:
        losses = estimate_loss()  # Estimate losses
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Sample a batch of training data
    xb, yb = get_batch('train')

    # Compute the loss for the batch
    logits, loss = model(xb, yb)  # Forward pass
    optimizer.zero_grad(set_to_none=True)  # Clear previous gradients
    loss.backward()  # Backpropagate the loss
    optimizer.step()  # Update model parameters

# End timer and print elapsed time
end_time = datetime.now()
elapse_time = (end_time - start_time).total_seconds()
print("Time taken: " + str(elapse_time) + " s")

# Save the trained model state to a file
torch.save(m.state_dict(), config["model_path"])

# Generate text using the trained model
context = torch.zeros((1, 1), dtype=torch.long, device=device)  # Initial context (empty input)
print(decode(m.generate(context, max_new_tokens=200)[0].tolist()))  # Generate and decode new tokens
