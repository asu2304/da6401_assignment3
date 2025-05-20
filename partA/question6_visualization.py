import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from torch.utils.data import DataLoader

from data_utils import TransliterationDataset, collate_fn, src_vocab, tgt_vocab
from model import Encoder, Decoder, Seq2Seq

# ─── Paths ───────────────────────────────────────────────────
BASE_DIR       = 'partA'
TEST_PATH      = os.path.join(BASE_DIR, 'lexicons', 'hi.translit.sampled.test.tsv')
CHKPT_ATTN     = os.path.join(BASE_DIR, 'model_use_attn.pt')

# ─── Device & token IDs ─────────────────────────────────────
device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pad_idx = tgt_vocab['<pad>']
sos_idx = tgt_vocab['<sos>']
eos_idx = tgt_vocab['<eos>']

# ─── Attempt to load a Devanagari font ──────────────────────
dev_font = None
# Linux path for Noto Sans Devanagari
linux_dev = '/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf'
# Windows common Devanagari font
win_dev   = 'C:/Windows/Fonts/Mangal.ttf'
if os.path.exists(linux_dev):
    dev_font = fm.FontProperties(fname=linux_dev)
elif os.path.exists(win_dev):
    dev_font = fm.FontProperties(fname=win_dev)

# ─── Invert vocabularies for ID→char lookup ─────────────────
inv_src = {v:k for k,v in src_vocab.items()}
inv_tgt = {v:k for k,v in tgt_vocab.items()}

# ─── 1) Infer model hyperparameters dynamically ─────────────
state = torch.load(CHKPT_ATTN, map_location='cpu')
emb_dim    = state['enc.emb.weight'].shape[1]
hh         = state['enc.rnn.weight_hh_l0']
hid_dim    = hh.shape[1]
num_gates  = hh.shape[0] // hid_dim
cell_type  = {1:'RNN',3:'GRU',4:'LSTM'}[num_gates]
enc_layers = len([k for k in state if k.startswith('enc.rnn.weight_ih_l')])
dec_layers = len([k for k in state if k.startswith('dec.rnn.weight_ih_l')])
w0         = state['dec.rnn.weight_ih_l0']
use_attn   = (w0.shape[1] == emb_dim + hid_dim)
dropout    = 0.2  # best-run value

# ─── 2) Build and load the attention model ─────────────────
enc   = Encoder(len(src_vocab), emb_dim, hid_dim, enc_layers, cell_type, dropout)
dec   = Decoder(len(tgt_vocab), emb_dim, hid_dim, dec_layers, cell_type, dropout, use_attn)
model = Seq2Seq(enc, dec, pad_idx, device).to(device)
model.load_state_dict(state)
model.eval()

# ─── 3) Prepare test loader (batch_size=1) ──────────────────
test_ds = TransliterationDataset(TEST_PATH, src_vocab, tgt_vocab)
test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

# ─── 4) Helper: run model & collect attention matrix ───────
def get_attention(src_tensor, max_len=64):
    """
    Args:
      src_tensor: [1, S_in] LongTensor
    Returns:
      attn_mat:   [L_out × S_in] numpy array
      pred_ids:   list of predicted token IDs (stops at <eos>)
    """
    with torch.no_grad():
        enc_out, enc_h = model.enc(src_tensor.to(device))
        dec_h         = model._init_decoder_hidden(enc_h)
        inp           = torch.tensor([sos_idx], device=device)
        attn_list, pred_ids = [], []
        for _ in range(max_len):
            logits, dec_h, attw = model.dec(inp, dec_h, enc_out)
            attn_list.append(attw[0].cpu().numpy())  # [S_in]
            top_id = logits.argmax(1).item()
            if top_id == eos_idx:
                break
            pred_ids.append(top_id)
            inp = torch.tensor([top_id], device=device)
    return np.stack(attn_list, axis=0), pred_ids

# ─── 5) Plot heatmaps for first 3 examples ──────────────────
sns.set_theme(style='whitegrid')  # lighter grid background
for idx, (src, tgt) in enumerate(test_dl):
    if idx >= 3:
        break

    # a) raw strings
    roman_str = test_ds.pairs[idx][0].split('<eos>')[0]
    attn_mat, pred_ids = get_attention(src)
    pred_str = ''.join(inv_tgt[i] for i in pred_ids).split('<eos>')[0]

    # b) character lists
    src_chars  = list(roman_str)
    out_chars  = list(pred_str)

    # c) slice matrix to actual lengths
    mat = attn_mat[:len(out_chars), :len(src_chars)]

    # d) plot
    plt.figure(
        figsize=(len(src_chars)*0.6 + 2,
                 len(out_chars)*0.6 + 2)
    )
    ax = sns.heatmap(
        mat,
        xticklabels=src_chars,
        yticklabels=out_chars,
        cmap='Blues',
        cbar_kws={'label':'Attention weight'},
        linewidths=0.3,
        linecolor='white',
        square=False
    )

    # e) annotate each cell with roman char
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(
                j, i,
                src_chars[j],
                ha='center', va='center',
                color='white' if mat[i,j] > 0.6 else 'black',
                fontsize=8
            )

    # f) set fonts so Devanagari labels render
    if dev_font:
        for lbl in ax.get_yticklabels():
            lbl.set_fontproperties(dev_font)

    ax.set_xlabel("Input (Roman) Characters")
    ax.set_ylabel("Predicted (Devanagari) Characters")
    ax.set_title(f"Attention Heatmap Example {idx+1}")

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    out_file = f"heatmap_{idx+1}.png"
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved {out_file}")
