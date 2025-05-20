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
BASE_DIR       = 'partA'
TEST_PATH      = os.path.join(BASE_DIR, 'lexicons', 'hi.translit.sampled.test.tsv')
CHKPT_ATTN     = os.path.join(BASE_DIR, 'model_use_attn.pt')

device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pad_idx  = tgt_vocab['<pad>']
sos_idx  = tgt_vocab['<sos>']
eos_idx  = tgt_vocab['<eos>']

# ─── Load Devanagari font ──────────────────────────────────────
dev_font = None
for path in [
    '/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf',  # Linux
    'C:/Windows/Fonts/Mangal.ttf'                                     # Windows
]:
    if os.path.exists(path):
        dev_font = fm.FontProperties(fname=path)
        break
if dev_font is None:
    raise RuntimeError("Devanagari font not found; install Noto Sans Devanagari or Mangal")

# ─── Infer hyperparameters & load model ─────────────────────────
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
dropout    = 0.2

enc = Encoder(len(src_vocab), emb_dim, hid_dim, enc_layers, cell_type, dropout)
dec = Decoder(len(tgt_vocab), emb_dim, hid_dim, dec_layers, cell_type, dropout, use_attn)
model = Seq2Seq(enc, dec, pad_idx, device).to(device)
model.load_state_dict(state)
model.eval()

# ─── Prepare test set (batch_size=1) ───────────────────────────
test_ds = TransliterationDataset(TEST_PATH, src_vocab, tgt_vocab)
test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

# ─── Helper to run model and collect attention ──────────────────
def get_attention(src_tensor, max_len=64):
    with torch.no_grad():
        enc_out, enc_h = model.enc(src_tensor.to(device))
        dec_h         = model._init_decoder_hidden(enc_h)
        inp           = torch.tensor([sos_idx], device=device)
        attn_list, pred_ids = [], []
        for _ in range(max_len):
            logits, dec_h, attw = model.dec(inp, dec_h, enc_out)
            attn_list.append(attw[0].cpu().numpy())
            top_id = logits.argmax(1).item()
            if top_id == eos_idx:
                break
            pred_ids.append(top_id)
            inp = torch.tensor([top_id], device=device)
    return np.stack(attn_list, axis=0), pred_ids

inv_tgt = {v:k for k,v in tgt_vocab.items()}

# ─── Plot 3×3 attention heatmaps with proper Devanagari labels ─
sns.set_theme(style='whitegrid')
os.makedirs('attention_heatmaps', exist_ok=True)

for idx, (src, tgt) in enumerate(test_dl):
    if idx >= 3:
        break

    # get attention weights and predicted IDs
    attn_mat, pred_ids = get_attention(src)
    roman_in = test_ds.pairs[idx][0].split('<eos>')[0]
    pred_str = ''.join(inv_tgt[i] for i in pred_ids).split('<eos>')[0]

    src_chars = list(roman_in)
    out_chars = list(pred_str)
    mat = attn_mat[:len(out_chars), :len(src_chars)]

    plt.figure(figsize=(len(src_chars)*0.6+2, len(out_chars)*0.6+2))
    ax = sns.heatmap(
        mat,
        cmap='YlOrBr',
        xticklabels=src_chars,
        yticklabels=out_chars,
        cbar_kws={'label':'Attention weight'},
        linewidths=0.3,
        linecolor='white'
    )

    # apply Devanagari font to y‐axis
    ax.set_yticklabels(out_chars, fontproperties=dev_font, rotation=0)
    ax.set_xticklabels(src_chars, rotation=45, ha='right')

    # annotate each cell with roman char (font fallback)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(
                j, i,
                src_chars[j],
                ha='center', va='center',
                color='white' if mat[i,j]>0.6 else 'black',
                fontsize=8
            )

    ax.set_xlabel("Input (Roman) Characters")
    ax.set_ylabel("Predicted (Devanagari) Characters", fontproperties=dev_font)
    ax.set_title(f"Attention Heatmap Example {idx+1}", fontproperties=dev_font)

    out_file = f'attention_heatmaps/heatmap_{idx+1}.png'
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved {out_file}")
