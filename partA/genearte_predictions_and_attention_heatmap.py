# # generate_attention_plots.py

# import os
# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# import seaborn as sns
# import matplotlib.font_manager as fm
# from torch.utils.data import DataLoader

# from data_utils import TransliterationDataset, collate_fn, src_vocab, tgt_vocab
# from model import Encoder, Decoder, Seq2Seq

# # ─── Paths & Device ──────────────────────────────────────────────
# BASE_DIR    = 'partA'
# TEST_PATH   = os.path.join(BASE_DIR, 'lexicons', 'hi.translit.sampled.test.tsv')
# CHKPT_ATTN  = os.path.join(BASE_DIR, 'model_use_attn.pt')

# device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# pad_idx  = tgt_vocab['<pad>']
# sos_idx  = tgt_vocab['<sos>']
# eos_idx  = tgt_vocab['<eos>']

# # ─── Load a Devanagari Font ──────────────────────────────────────
# dev_font = None
# linux_dev = '/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf'
# win_dev   = 'C:/Windows/Fonts/Mangal.ttf'
# for path in (linux_dev, win_dev):
#     if os.path.exists(path):
#         dev_font = fm.FontProperties(fname=path)
#         break
# if dev_font is None:
#     print("⚠️  No Devanagari font found; install Noto Sans Devanagari or Mangal.")

# # ─── Invert the target vocab for ID→Devanagari lookup ─────────────
# inv_tgt = {idx:char for char, idx in tgt_vocab.items()}

# # ─── Dynamically infer hyperparameters & load model ──────────────
# state     = torch.load(CHKPT_ATTN, map_location='cpu')
# emb_dim   = state['enc.emb.weight'].shape[1]
# hh        = state['enc.rnn.weight_hh_l0']
# hid_dim   = hh.shape[1]
# num_gates = hh.shape[0] // hid_dim
# cell_map  = {1:'RNN',3:'GRU',4:'LSTM'}
# cell_type = cell_map[num_gates]
# enc_layers= len([k for k in state if k.startswith('enc.rnn.weight_ih_l')])
# dec_layers= len([k for k in state if k.startswith('dec.rnn.weight_ih_l')])
# w0        = state['dec.rnn.weight_ih_l0']
# use_attn  = (w0.shape[1] == emb_dim + hid_dim)

# enc = Encoder(len(src_vocab), emb_dim, hid_dim, enc_layers, cell_type, dropout=0.0)
# dec = Decoder(len(tgt_vocab), emb_dim, hid_dim, dec_layers, cell_type,
#               dropout=0.0, use_attn=use_attn)
# model = Seq2Seq(enc, dec, pad_idx, device).to(device)
# model.load_state_dict(state)
# model.eval()

# # ─── Prepare the test set (batch_size=1 for per-example plotting) ─
# test_ds = TransliterationDataset(TEST_PATH, src_vocab, tgt_vocab)
# test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

# # ─── Helper to run greedy decoding and collect attention ──────────
# def get_attention(src_tensor, max_len=64):
#     with torch.no_grad():
#         enc_out, enc_h = model.enc(src_tensor.to(device))
#         dec_h         = model._init_decoder_hidden(enc_h)
#         inp           = torch.tensor([sos_idx], device=device)
#         attn_list     = []
#         pred_ids      = []
#         for _ in range(max_len):
#             logits, dec_h, attw = model.dec(inp, dec_h, enc_out)
#             attn_list.append(attw[0].cpu().numpy())  # shape: [S_in]
#             top_id = logits.argmax(1).item()
#             if top_id == eos_idx:
#                 break
#             pred_ids.append(top_id)
#             inp = torch.tensor([top_id], device=device)
#     return np.stack(attn_list, axis=0), pred_ids

# # ─── Plot a 3×3 grid of attention heatmaps ───────────────────────
# sns.set_theme(style='whitegrid')
# fig, axes = plt.subplots(3, 3, figsize=(12,12))

# for idx, ((src, tgt), ax) in enumerate(zip(test_dl, axes.flat)):
#     if idx >= 9:
#         break

#     # original Roman input & run model
#     roman_in = test_ds.pairs[idx][0].split('<eos>')[0]
#     attn_mat, pred_ids = get_attention(src)

#     # decode Devanagari prediction
#     pred_str = ''.join(inv_tgt[i] for i in pred_ids)

#     # character axes
#     src_chars = list(roman_in)
#     tgt_chars = list(pred_str)

#     # crop matrix to actual lengths
#     mat = attn_mat[:len(tgt_chars), :len(src_chars)]

#     # seaborn heatmap
#     sns.heatmap(
#         mat,
#         ax=ax,
#         cmap='YlOrBr',
#         xticklabels=src_chars,
#         yticklabels=tgt_chars,
#         cbar=(idx==0),
#         cbar_kws={'label':'Attention weight'},
#         linewidths=0.3,
#         linecolor='white'
#     )

#     # force Devanagari font on y-ticks (and x if desired)
#     if dev_font:
#         for lbl in ax.get_yticklabels():
#             lbl.set_fontproperties(dev_font)
#         for lbl in ax.get_xticklabels():
#             lbl.set_fontproperties(dev_font)

#     ax.set_xlabel('in')
#     ax.set_ylabel('predicted out')
#     ax.set_title(f'Example {idx+1}')

# plt.tight_layout()
# os.makedirs('attention_heatmaps', exist_ok=True)
# fig.savefig('attention_heatmaps/heatmaps_3x3.png', dpi=300)
# print("✅ Saved attention_heatmaps/heatmaps_3x3.png")

# generate_attention_plots.py
# ------------------------------

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
from torch.utils.data import DataLoader

from data_utils import TransliterationDataset, collate_fn, src_vocab, tgt_vocab
from model import Encoder, Decoder, Seq2Seq

# ─── Paths & Device ──────────────────────────────────────────────
BASE_DIR    = 'partA'
TEST_PATH   = os.path.join(BASE_DIR, 'lexicons', 'hi.translit.sampled.test.tsv')
CHKPT_ATTN  = os.path.join(BASE_DIR, 'model_use_attn.pt')

device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pad_idx  = tgt_vocab['<pad>']
sos_idx  = tgt_vocab['<sos>']
eos_idx  = tgt_vocab['<eos>']

# ─── Load a Devanagari Font ──────────────────────────────────────
dev_font = None
linux_dev = '/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf'
win_dev   = 'C:/Windows/Fonts/Mangal.ttf'
for path in (linux_dev, win_dev):
    if os.path.exists(path):
        dev_font = fm.FontProperties(fname=path)
        break
if dev_font is None:
    print("⚠️  No Devanagari font found; install Noto Sans Devanagari or Mangal.")

# ─── Invert the target vocab for ID→Devanagari lookup ─────────────
inv_tgt = {idx:char for char, idx in tgt_vocab.items()}

# ─── Dynamically infer hyperparameters & load model ──────────────
state     = torch.load(CHKPT_ATTN, map_location='cpu')
emb_dim   = state['enc.emb.weight'].shape[1]
hh        = state['enc.rnn.weight_hh_l0']
hid_dim   = hh.shape[1]
num_gates = hh.shape[0] // hid_dim
cell_map  = {1:'RNN',3:'GRU',4:'LSTM'}
cell_type = cell_map[num_gates]
enc_layers= len([k for k in state if k.startswith('enc.rnn.weight_ih_l')])
dec_layers= len([k for k in state if k.startswith('dec.rnn.weight_ih_l')])
w0        = state['dec.rnn.weight_ih_l0']
use_attn  = (w0.shape[1] == emb_dim + hid_dim)
dropout   = 0.2

enc = Encoder(len(src_vocab), emb_dim, hid_dim, enc_layers, cell_type, dropout)
dec = Decoder(len(tgt_vocab), emb_dim, hid_dim, dec_layers, cell_type,
              dropout, use_attn=use_attn)
model = Seq2Seq(enc, dec, pad_idx, device).to(device)
model.load_state_dict(state)
model.eval()

# ─── Prepare the test set (batch_size=1) ─────────────────────────
test_ds = TransliterationDataset(TEST_PATH, src_vocab, tgt_vocab)
test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

# ─── Helper to run greedy decoding and collect attention ──────────
def get_attention(src_tensor, max_len=64):
    with torch.no_grad():
        enc_out, enc_h = model.enc(src_tensor.to(device))
        dec_h         = model._init_decoder_hidden(enc_h)
        inp           = torch.tensor([sos_idx], device=device)
        attn_list     = []
        pred_ids      = []
        for _ in range(max_len):
            logits, dec_h, attw = model.dec(inp, dec_h, enc_out)
            attn_list.append(attw[0].cpu().numpy())  # shape: [S_in]
            top_id = logits.argmax(1).item()
            if top_id == eos_idx:
                break
            pred_ids.append(top_id)
            inp = torch.tensor([top_id], device=device)
    return np.stack(attn_list, axis=0), pred_ids

# ─── Plot a 3×3 grid of attention heatmaps + track accuracy ─────
sns.set_theme(style='whitegrid')
fig, axes = plt.subplots(3, 3, figsize=(12,12))

correct_chars = 0   # ← initialize counters
total_chars   = 0
correct_words = 0
total_words   = 0

for idx, ((src, tgt), ax) in enumerate(zip(test_dl, axes.flat)):
    if idx >= 9:
        break

    # roman input and attention
    roman_in = test_ds.pairs[idx][0].split('<eos>')[0]
    attn_mat, pred_ids = get_attention(src)

    # decode prediction to string
    pred_str = ''.join(inv_tgt[i] for i in pred_ids)

    # axes labels
    src_chars = list(roman_in)
    tgt_chars = list(pred_str)

    # crop matrix to true lengths
    mat = attn_mat[:len(tgt_chars), :len(src_chars)]

    sns.heatmap(
        mat,
        ax=ax,
        cmap='YlOrBr',  # yellow–orange theme
        xticklabels=src_chars,
        yticklabels=tgt_chars,
        cbar=(idx==0),
        cbar_kws={'label':'Attention weight'},
        linewidths=0.3,
        linecolor='white'
    )

    # apply Devanagari font to y‐labels
    if dev_font:
        for lbl in ax.get_yticklabels():
            lbl.set_fontproperties(dev_font)

    ax.set_xlabel('Input (Roman)')
    ax.set_ylabel('Predicted (Devanagari)')
    ax.set_title(f'Example {idx+1}')

    # ─── Compute character‐level accuracy ─────────────────────
    ref_str = test_ds.pairs[idx][1].split('<eos>')[0]
    for c_ref, c_pred in zip(ref_str, pred_str):
        if c_ref == c_pred:
            correct_chars += 1
    total_chars += len(ref_str)

    # ─── Compute word‐level accuracy ─────────────────────────
    if pred_str == ref_str:
        correct_words += 1
    total_words += 1

plt.tight_layout()
os.makedirs('attention_heatmaps', exist_ok=True)
fig.savefig('attention_heatmaps/heatmaps_3x3.png', dpi=300)
print("✅ Saved attention_heatmaps/heatmaps_3x3.png")

# ─── Print final test accuracies ─────────────────────────────
char_acc = (correct_chars / total_chars)*100 if total_chars>0 else 0
word_acc = (correct_words / total_words)*100 if total_words>0 else 0
print(f"Test Character-level Accuracy: {char_acc:.2f}%")
print(f"Test Word-level Accuracy: {word_acc:.2f}%")
