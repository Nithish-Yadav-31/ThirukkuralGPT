import torch
import torch.nn as nn
from torch.nn import functional as F
import json

from bigram import BigramLanguageModel  # Import the BigramLanguageModel class

# Load configuration settings from a JSON file
config = {}
with open("config.json", "r") as jsonfile:
    config = json.load(jsonfile)  # Read the configuration into a dictionary
    print("Config: ", config)  # Print the loaded configuration

# Determine the computation device (CPU or GPU)
device = 'cuda:0'  # Default to using the first CUDA device
if config["allow_gpu"]:  # Check if GPU usage is permitted
    if torch.cuda.is_available():  # If CUDA is available, use it
        device = 'cuda' 
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():  # Check for Metal Performance Shaders (Apple Silicon)
        device = 'mps' 
print("Using device: " + device)  # Output the selected device

# Load text data from the specified file path in the configuration
with open(config["data_path"], 'r', encoding='utf-8') as f:
    text = f.read()  # Read the entire content of the text file

# Identify unique characters in the text
chars = sorted(list(set(text)))  # Create a sorted list of unique characters
vocab_size = len(chars)  # Determine the vocabulary size based on unique characters

# Create mappings between characters and integers
stoi = { ch: i for i, ch in enumerate(chars) }  # Map characters to indices (string to index)
itos = { i: ch for i, ch in enumerate(chars) }  # Map indices to characters (index to string)

# Define encoding and decoding functions
encode = lambda s: [stoi[c] for c in s]  # Encoder: converts a string to a list of integers
decode = lambda l: ''.join([itos[i] for i in l])  # Decoder: converts a list of integers back to a string

# Initialize the Bigram Language Model
model = BigramLanguageModel(device=device, config=config, vocab_size=vocab_size)
m = model.to(device)  # Move the model to the selected device

# Load pre-trained model weights from a file
m.load_state_dict(torch.load(config["model_path"]))  # Load the saved state dictionary

# Print the total number of parameters in the model
print(sum(p.numel() for p in m.parameters()) / 1e6, 'M parameters')  # Output the number of model parameters in millions

# Generate text using the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)  # Create initial context tensor (empty input)

generated_text = decode(m.generate(context, max_new_tokens=100)[0].tolist())  # Generate 100 new tokens and decode

# Write the generated text to output.txt
with open("output.txt", "w", encoding='utf-8') as output_file:
    output_file.write(generated_text)  # Write the generated text to the file
    
# Generate text and decode the output into a readable string
print(decode(m.generate(context, max_new_tokens=100)[0].tolist()))  # Generate and print 100 new tokens
