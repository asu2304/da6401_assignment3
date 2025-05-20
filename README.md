# Transliteration System with Attention Mechanisms

This repository contains an implementation of a Seq2Seq neural machine translation system for transliterating between Roman script and Devanagari script. The system includes both vanilla Seq2Seq and attention-based models.

## Repository Structure

The repository is organized as follows:

```
Transliteration_System/
├── attention_heatmaps/           # Directory containing attention visualization heatmaps
├── partA/                        # Main code directory
│   ├── lexicons/                 # Dataset directory with train/dev/test splits
│   ├── beam_search.py            # Implementation of beam search decoding
│   ├── data_utils.py             # Dataset processing utilities
│   ├── model.py                  # Seq2Seq model architecture definition
│   ├── model_use_attn.pt         # Trained model weights (with attention)
│   ├── model_without_attn.pt     # Trained model weights (without attention)
│   ├── predictions_*.txt         # Model outputs on test set
│   ├── question4_grid_*.py       # Visualization for comparing models
│   ├── question6_visualization.py # Connectivity visualization
│   ├── requirements.txt          # Python dependencies
│   └── train.py                  # Training script
├── comparison_grid.png           # Visual comparison of model outputs
├── heatmap_*.png                 # Attention visualization heatmaps
└── attention_fixed_examples.csv  # Examples where attention corrects vanilla model errors
```

## Setup & Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/asu2304/da6401_assignment3
   cd da6401_assignment3
   ```

2. Install required packages:
   ```bash
   pip install -r partA/requirements.txt
   ```

## Usage Instructions

### Training Models

To train with wandb sweep:

```bash
cd partA
python train.py
```

Here in this script i have implemented with and without attention mechanism in single code and passed use_attention as a hyperparameter so you can train both the vanilla Seq2Seq model and the attention-based model using the hyperparameters defined in the script. Model checkpoints will be saved accordingly.

### Generating Predictions and Evaluation on test set

To generate predictions on the test set using the pre-trained models you can run the follwing jupter notebook: 
```
partA\prediction_notebook_with_attention.ipynb
partA\prediction_notebook_without_attention.ipynb
```

This will:
1. Load both trained models (with and without attention)
2. Run inference on the test set
3. Save predictions to `predictions_with_attention.txt` and `predictions_without_attention.txt`
4. Print character-level and word-level accuracies for both models

### Visualizing Attention Mechanisms

#### Attention Heatmaps

To generate attention heatmaps that show which input characters the model focuses on:

```bash
cd partA
python genearte_attention_heatmap.py
```

This creates heatmap visualizations for three test examples, showing the attention weights between input and output characters.

#### Connectivity Visualization

To create "connectivity" plots showing which input character is most attended to at each output step:

```bash
cd partA
python question6_visualization.py
```

#### Compare Model Outputs

To generate a visual comparison of outputs from both models:

```bash
cd partA
python question4_grid_visualization.py
```

This creates `comparison_grid.png`, highlighting where the attention model succeeds and the vanilla model fails.




## Model Architecture

- **Encoder**: Bi-directional LSTM/GRU network that encodes the input sequence
- **Decoder**: LSTM/GRU network that generates the output sequence
- **Attention Mechanism**: Optional attention layer that helps the decoder focus on relevant parts of the input

## Hyperparameters

### Best Vanilla Model:
- Embedding dimension: 32
- Hidden dimension: 256
- Encoder layers: 3
- Decoder layers: 3
- Cell type: LSTM
- Dropout: 0.2
  
### Best Attention Model:
- batch_size: 128
- beam_size: 3
- cell_type: GRU
- dec_layers: 1
- dropout: 0.2
- emb_dim: 16
- enc_layers: 3
- hid_dim: 256
- lr: 0.00



## Results

The attention-based model significantly outperforms the vanilla model:
- **With Attention**: Higher character and word-level accuracy
- **Without Attention**: More errors, particularly with long sequences and complex character mappings

Specifically, attention improves transliteration by helping maintain correct mappings for:
1. Long sequences
2. Consonant clusters
3. Diacritics and special characters
4. Vowel length distinctions

The visualizations in `attention_heatmaps/` and `heatmap_*.png` illustrate how the attention mechanism learns to focus on the appropriate input characters when generating each output character.

