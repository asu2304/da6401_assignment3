

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data_utils import TransliterationDataset, collate_fn, src_vocab, tgt_vocab
from model import Encoder, Decoder, Seq2Seq

# Fixed hyperparameters (without wandb)
batch_size = 128
beam_size = 1
cell_type = 'LSTM'
dec_layers = 1
dropout = 0.2
emb_dim = 256
enc_layers = 3
hid_dim = 256
lr = 0.001
use_attn = True

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, 'lexicons', 'hi.translit.sampled.train.tsv')
VAL_PATH = os.path.join(BASE_DIR, 'lexicons', 'hi.translit.sampled.dev.tsv')
TEST_PATH = os.path.join(BASE_DIR, 'lexicons', 'hi.translit.sampled.test.tsv')
MODEL_PATH = os.path.join(BASE_DIR, 'trained_attn_model.pt')

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pad_idx = tgt_vocab['<pad>']
eos_idx = tgt_vocab['<eos>']

# Load datasets
train_ds = TransliterationDataset(TRAIN_PATH, src_vocab, tgt_vocab)
val_ds = TransliterationDataset(VAL_PATH, src_vocab, tgt_vocab)
test_ds = TransliterationDataset(TEST_PATH, src_vocab, tgt_vocab)

train_dl = DataLoader(train_ds, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=batch_size, collate_fn=collate_fn)
test_dl = DataLoader(test_ds, batch_size=batch_size, collate_fn=collate_fn)

# Build model
print(f"Building model with parameters:")
print(f"  emb_dim: {emb_dim}, hid_dim: {hid_dim}")
print(f"  enc_layers: {enc_layers}, dec_layers: {dec_layers}")
print(f"  cell_type: {cell_type}, dropout: {dropout}")
print(f"  use_attn: {use_attn}")

enc = Encoder(len(src_vocab), emb_dim, hid_dim, enc_layers, cell_type, dropout)
dec = Decoder(len(tgt_vocab), emb_dim, hid_dim, dec_layers, cell_type, dropout, use_attn)
model = Seq2Seq(enc, dec, pad_idx, device).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training function
def train_epoch(model, dataloader, optimizer, criterion, pad_idx):
    model.train()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    
    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        
        output = model(src, tgt[:, :-1])  # teacher forcing
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        tgt_y = tgt[:, 1:].contiguous().view(-1)
        
        loss = criterion(output, tgt_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * tgt_y.size(0)
        preds = output.argmax(1)
        correct = (preds == tgt_y).sum().item()
        total_correct += correct
        total_tokens += (tgt_y != pad_idx).sum().item()
    
    return total_loss / total_tokens, total_correct / total_tokens

# Validation/Evaluation function
def eval_epoch(model, dataloader, criterion, pad_idx):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    
    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            
            output = model(src, tgt[:, :-1])
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            tgt_y = tgt[:, 1:].contiguous().view(-1)
            
            loss = criterion(output, tgt_y)
            total_loss += loss.item() * tgt_y.size(0)
            preds = output.argmax(1)
            correct = (preds == tgt_y).sum().item()
            total_correct += correct
            total_tokens += (tgt_y != pad_idx).sum().item()
    
    return total_loss / total_tokens, total_correct / total_tokens

# Test using greedy decoding and calculate accuracy
def evaluate_test_set(model, dataloader):
    model.eval()
    all_preds = []
    all_refs = []
    
    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            B, T = tgt.size()
            
            # Get references
            for i in range(B):
                ref_seq = []
                for idx in tgt[i][1:].cpu().tolist():  # skip <sos>
                    if idx == eos_idx:
                        break
                    ref_seq.append(next(ch for ch,v in tgt_vocab.items() if v==idx))
                all_refs.append(''.join(ref_seq))
            
            # Encode
            enc_out, enc_h = model.enc(src)
            dec_h = model._init_decoder_hidden(enc_h)
            inp = tgt[:, 0]  # <sos>
            
            # Decode greedily
            preds = torch.zeros(B, T, dtype=torch.long, device=device)
            for t in range(1, T):
                logits, dec_h, _ = model.dec(
                    inp, dec_h,
                    enc_out if model.dec.use_attn else None
                )
                top1 = logits.argmax(1)
                preds[:, t] = top1
                inp = top1
            
            # Convert to strings
            for i in range(B):
                pred_seq = []
                for idx in preds[i].cpu().tolist():
                    if idx == eos_idx:
                        break
                    pred_seq.append(next(ch for ch,v in tgt_vocab.items() if v==idx))
                all_preds.append(''.join(pred_seq))
    
    # Calculate character-level accuracy
    char_matches = 0
    total_chars = 0
    for pred, ref in zip(all_preds, all_refs):
        for p_char, r_char in zip(pred, ref):
            if p_char == r_char:
                char_matches += 1
        total_chars += len(ref)
    
    # Calculate word-level accuracy
    word_matches = sum(p == r for p, r in zip(all_preds, all_refs))
    total_words = len(all_refs)
    
    char_acc = (char_matches / total_chars) * 100 if total_chars > 0 else 0
    word_acc = (word_matches / total_words) * 100 if total_words > 0 else 0
    
    return char_acc, word_acc, all_preds

# Train for 10 epochs
epochs = 10
best_val_loss = float('inf')

for epoch in range(1, epochs+1):
    train_loss, train_acc = train_epoch(model, train_dl, optimizer, criterion, pad_idx)
    val_loss, val_acc = eval_epoch(model, val_dl, criterion, pad_idx)
    
    print(f'Epoch {epoch:2d}: Train Loss={train_loss:.4f}, Train Acc={train_acc*100:.2f}%, Val Loss={val_loss:.4f}, Val Acc={val_acc*100:.2f}%')
    
    # Save the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), MODEL_PATH)
        print(f'  ← New best model saved to {MODEL_PATH}')

# Load the best model for evaluation
print("\nLoading best model for evaluation...")
model.load_state_dict(torch.load(MODEL_PATH))

# Evaluate on the test set
print("\nEvaluating on test set...")
char_acc, word_acc, test_preds = evaluate_test_set(model, test_dl)

print("\n" + "="*60)
print("TEST ACCURACY RESULTS")
print("="*60)
print(f"Character-level accuracy: {char_acc:.2f}%")
print(f"Word-level accuracy:      {word_acc:.2f}%")
print("="*60)

# Save predictions
with open('predictions_with_attention.txt', 'w', encoding='utf-8') as f:
    for pred in test_preds:
        f.write(pred + '\n')

print("Predictions saved to predictions_with_attention.txt")
