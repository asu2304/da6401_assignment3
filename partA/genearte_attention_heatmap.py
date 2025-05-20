# plot_attention_heatmaps.py

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader

from data_utils import TransliterationDataset, collate_fn, src_vocab, tgt_vocab
from model import Encoder, Decoder, Seq2Seq

# ─── Paths ───────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
TEST_PATH       = os.path.join(BASE_DIR, 'lexicons', 'hi.translit.sampled.test.tsv')
CHKPT_PATH      = os.path.join(BASE_DIR, 'model_use_attn.pt')
PRED_ATTN_FILE  = os.path.join(BASE_DIR, 'predictions_with_attention.txt')

# ─── Device & tokens ─────────────────────────────────────────
device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pad_idx = tgt_vocab['<pad>']
sos_idx = tgt_vocab['<sos>']
eos_idx = tgt_vocab['<eos>']

# ─── Reverse vocab lookup ────────────────────────────────────
inv_src = {v:k for k,v in src_vocab.items()}
inv_tgt = {v:k for k,v in tgt_vocab.items()}

# ─── 1) Infer hyperparameters from checkpoint ────────────────
state = torch.load(CHKPT_PATH, map_location='cpu')
emb_dim   = state['enc.emb.weight'].shape[1]
hh        = state['enc.rnn.weight_hh_l0']
hid_dim   = hh.shape[1]
num_gates = hh.shape[0] // hid_dim
cell_type = {1:'RNN',3:'GRU',4:'LSTM'}[num_gates]
enc_layers= len([k for k in state if k.startswith('enc.rnn.weight_ih_l')])
dec_layers= len([k for k in state if k.startswith('dec.rnn.weight_ih_l')])
# detect attention
w0        = state['dec.rnn.weight_ih_l0']
use_attn  = (w0.shape[1] == emb_dim + hid_dim)
dropout   = 0.2  # per your best run

# ─── 2) Build and load model ─────────────────────────────────
enc = Encoder(len(src_vocab), emb_dim,   hid_dim, enc_layers, cell_type, dropout)
dec = Decoder(len(tgt_vocab), emb_dim,   hid_dim, dec_layers, cell_type, dropout, use_attn)
model = Seq2Seq(enc, dec, pad_idx, device).to(device)
model.load_state_dict(state)
model.to(device).eval()

# ─── 3) Read predicted outputs ────────────────────────────────
with open(PRED_ATTN_FILE, encoding='utf-8') as f:
    predictions = [line.strip() for line in f]

# ─── 4) Prepare test‐set DataLoader ──────────────────────────
test_ds = TransliterationDataset(TEST_PATH, src_vocab, tgt_vocab)
test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

# ─── 5) Helper to get attention matrix per example ───────────
def get_attention(src, tgt):
    """Returns (attn_mat, pred_ids) for a single example."""
    with torch.no_grad():
        enc_out, enc_h = model.enc(src.to(device))
        dec_h = model._init_decoder_hidden(enc_h)
        inp   = torch.tensor([sos_idx], device=device)
        attn_list, pred_ids = [], []
        for _ in range(100):
            logits, dec_h, attw = model.dec(inp, dec_h, enc_out)
            attn_list.append(attw[0].cpu().numpy())  # shape=[S]
            top = logits.argmax(1).item()
            if top == eos_idx:
                break
            pred_ids.append(top)
            inp = torch.tensor([top], device=device)
    return np.stack(attn_list, axis=0), pred_ids

# ─── 6) Plot heatmaps for first 3 examples ───────────────────
sns.set()  # seaborn theme
for idx, (src, tgt) in enumerate(test_dl):
    if idx >= 3: break

    # raw roman input and predicted output
    src_str  = test_ds.pairs[idx][0]
    pred_str = predictions[idx]

    # get attention & predicted IDs
    attn_mat, pred_ids = get_attention(src, tgt)

    # truncate at <eos> if present
    if '<eos>' in src_str:  src_str  = src_str.split('<eos>')[0]
    if '<eos>' in pred_str: pred_str = pred_str.split('<eos>')[0]

    src_chars  = list(src_str)
    pred_chars = list(pred_str)

    # slice matrix to actual lengths
    H, W = attn_mat.shape
    mat = attn_mat[:len(pred_chars), :len(src_chars)]

    # plot
    plt.figure(figsize=(len(src_chars)*0.6+2, len(pred_chars)*0.6+2))
    ax = sns.heatmap(
        mat,
        xticklabels=src_chars,
        yticklabels=pred_chars,
        cmap='magma',
        cbar_kws={'label':'Attention weight'},
        linewidths=0.5,
        linecolor='white'
    )

    # annotate each cell with its roman input char
    for i in range(len(pred_chars)):
        for j in range(len(src_chars)):
            ax.text(
                j, i,
                src_chars[j],
                ha='center', va='center',
                color='white' if mat[i,j]>0.5 else 'black',
                fontsize=8
            )

    ax.set_xlabel("Input (Roman) Characters")
    ax.set_ylabel("Predicted (Devanagari) Characters")
    ax.set_title(f"Attention Heatmap Example {idx+1}")
    plt.tight_layout()

    out_file = f"heatmap_{idx+1}.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved {out_file}")
