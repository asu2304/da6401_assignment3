# evaluate_and_plot.py

import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from data_utils import TransliterationDataset, collate_fn, src_vocab, tgt_vocab
from model import Encoder, Decoder, Seq2Seq

# Paths
TEST_PATH         = 'C:\Users\Ashutosh Patidar\OneDrive\Documents\GitHub\Transliteration_System\partA\lexicons\hi.translit.sampled.test.tsv'
MODEL_ATTN_PATH   = '/kaggle/input/da6401_transliteration_trained/pytorch/default/1/model_use_attn.pt'
MODEL_NO_ATTN_PATH= '/kaggle/input/da6401_transliteration_trained/pytorch/default/1/model_without_attn.pt'

# Device and pad index
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pad_idx = tgt_vocab['<pad>']

# Best‐run hyperparameters (must match saved checkpoint)
best_emb_dim    = 256
best_hid_dim    = 256
best_enc_layers = 1
best_dec_layers = 1
best_cell_type  = 'LSTM'
best_dropout    = 0.3

# 1. Prepare test DataLoader
test_dataset = TransliterationDataset(TEST_PATH, src_vocab, tgt_vocab)
test_loader  = DataLoader(test_dataset, batch_size=128, collate_fn=collate_fn)

# 2. Instantiate and load both models
def load_model(path, use_attn):
    enc = Encoder(len(src_vocab), best_emb_dim, best_hid_dim,
                  best_enc_layers, best_cell_type, best_dropout)
    dec = Decoder(len(tgt_vocab), best_emb_dim, best_hid_dim,
                  best_dec_layers, best_cell_type, best_dropout,
                  use_attn=use_attn)
    model = Seq2Seq(enc, dec, pad_idx, device).to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

model_no_attn = load_model(MODEL_NO_ATTN_PATH, use_attn=False)
model_attn    = load_model(MODEL_ATTN_PATH,    use_attn=True)

# Ensure output directories
os.makedirs('predictions_vanilla', exist_ok=True)
os.makedirs('predictions_attention', exist_ok=True)
os.makedirs('attention_heatmaps',    exist_ok=True)

# 3. Greedy‐decode and save predictions
def save_predictions(model, loader, out_path, collect_attn=False):
    """
    If collect_attn==False, just greedy‐decode.
    If collect_attn==True, returns (preds, attn_matrices) for each example.
    """
    all_preds = []
    all_attn  = []  # list of [T_out × T_in] arrays

    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            B, T = tgt.size()
            # run encoder
            enc_out, enc_hidden = model.enc(src)
            # init decoder hidden
            dec_hidden = model._init_decoder_hidden(enc_hidden)
            inp = tgt[:,0]  # <sos>

            # buffers for this batch
            batch_preds = torch.zeros(B, T, dtype=torch.long, device=device)
            batch_attn  = []

            for t in range(1, T):
                pred, dec_hidden, attn_w = model.dec(
                    inp, dec_hidden,
                    enc_out if model.dec.use_attn else None
                )
                batch_preds[:,t] = pred.argmax(1)
                inp = batch_preds[:,t]

                if collect_attn:
                    # attn_w: [B, T_in]       → store per‐example
                    batch_attn.append(attn_w.cpu())

            # collect strings & attention
            for i in range(B):
                # decode to chars (stop at <eos>)
                seq = []
                for idx in batch_preds[i].cpu().tolist():
                    if idx == tgt_vocab['<eos>']: break
                    seq.append(next(k for k,v in tgt_vocab.items() if v==idx))
                all_preds.append(''.join(seq))

                if collect_attn:
                    # stack [T_out × T_in]
                    attn_matrix = torch.stack([step[i] for step in batch_attn], dim=0)
                    all_attn.append(attn_matrix.numpy())

    # write to file
    with open(out_path, 'w', encoding='utf-8') as f:
        for line in all_preds:
            f.write(line + '\n')

    return all_attn

# Vanilla predictions
_ = save_predictions(
    model_no_attn, test_loader,
    out_path='predictions_vanilla/predictions_vanilla.txt',
    collect_attn=False
)

# Attention predictions + collect heatmaps
attn_matrices = save_predictions(
    model_attn, test_loader,
    out_path='predictions_attention/predictions_attention.txt',
    collect_attn=True
)

# 4. Plot a 3×3 grid of attention heatmaps for the first 9 examples
for idx in range(9):
    attn = attn_matrices[idx]  # shape [T_out, T_in]
    # extract corresponding source & target strings
    src, tgt = test_dataset[idx]
    src_chars = [next(k for k,v in src_vocab.items() if v==i) for i in src.tolist()]
    tgt_chars = [next(k for k,v in tgt_vocab.items() if v==i) for i in tgt.tolist()[1:]]  # skip <sos>
    # clip at <eos>
    if '<eos>' in src_chars: src_chars = src_chars[:src_chars.index('<eos>')+1]
    if '<eos>' in tgt_chars: tgt_chars = tgt_chars[:tgt_chars.index('<eos>')]

    plt.subplot(3, 3, idx+1)
    plt.imshow(attn, cmap='viridis')
    plt.xticks(range(len(src_chars)), src_chars, rotation=90)
    plt.yticks(range(len(tgt_chars)), tgt_chars)
    plt.xlabel('Input chars'); plt.ylabel('Output chars')
    plt.title(f'Example {idx+1}')

plt.tight_layout()
plt.savefig('attention_heatmaps/heatmaps_3x3.png')
print("Done. Predictions and heatmaps saved.")
