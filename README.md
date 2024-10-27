# ThirukkuralGPT

ThirukkuralGPT is a simple implementation of a Bigram Language Model using PyTorch. It trains on text data and generates new text based on the learned patterns.

## Contents

- `train.py`: Script to train the Bigram Language Model.
- `generate.py`: Script to generate text using the trained model.
- `bigram.py`: Contains the `BigramLanguageModel` class definition.
- `prepare.py`: Script to prepare the dataset for training the model.
- `config.json`: Configuration file for model training and generation settings.

## Requirements

- Python 3.10+
- PyTorch 2.0+

## GPU Requirement

- This model had 42M parameters, and was trained on a RTX 4060, and it barely took just 30 to 40 mins to train.

## Setup

1. **Install Dependencies**: Make sure to install the required packages. You can do this using pip:

   ```bash
   pip install torch
   ```

2. **Prepare Data**: Ensure you have a text file (`kural.txt`) with your training data. Adjust the `data_path` in `config.json` if necessary.

3. **Configure Settings**: Edit `config.json` to set your training parameters and file paths.

## Training the Model

To train the model, run the following command:

```bash
python train.py
```

This will start the training process and save the model to the specified path in `config.json`.

## Generating Text

After training, you can generate text using the trained model by running:

```bash
python generate.py
```


## Sample output would look like

```bash

உவப்பத்தடின் ஒப்பம் கழியில் பாவாயின் மாற்றார்க்கு
இனனிலனொடு ஐந்தும் இடத்து.

வினைபகை என்றிரண்டின் எச்சம் நினையுங்கால்
தீயெச்சம் போலத் தெறும்.

பிறன்பொருளைக் கண்ணார் அறிவறிந்து
வேண்டுப வேட்பச் சொலல்.

பழுத்தும் கொளல்வேண்டும் மன்ற அடுத்தும்
பிறன்போல நிற்கும் பழி.
```

The generated text will be saved to `output.txt` and also printed to the console.

## Configuration

The configuration file (`config.json`) contains the following parameters:

- `allow_gpu`: Set to `true` to enable GPU support if available.
- `batch_size`: Number of samples processed in one iteration.
- `block_size`: Size of the input sequences.
- `max_iters`: Maximum number of training iterations.
- `learning_rate`: Learning rate for the optimizer.
- `eval_interval`: Frequency of evaluating the model during training.
- `eval_iters`: Number of iterations for evaluation.
- `n_embd`: Dimension of the embedding layer.
- `n_head`: Number of attention heads.
- `n_layer`: Number of layers in the model.
- `dropout`: Dropout rate for regularization.
- `model_path`: Path to save the trained model.
- `data_path`: Path to the training data.
