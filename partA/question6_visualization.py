# question6_visualization.py

import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from data_utils import TransliterationDataset, collate_fn, src_vocab, tgt_vocab
from model import Encoder, Decoder, Seq2Seq

# ─── Paths ───────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
TEST_PATH = 'partA\lexicons\hi.translit.sampled.test.tsv'
CHKPT     = os.path.join(BASE_DIR, 'model_use_attn.pt')

# ─── Device & special token indices ─────────────────────────
device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pad_idx = tgt_vocab['<pad>']
sos_idx = tgt_vocab['<sos>']
eos_idx = tgt_vocab['<eos>']

# ─── Reverse‐lookup for ID→char ─────────────────────────────
inv_src = {v:k for k,v in src_vocab.items()}
inv_tgt = {v:k for k,v in tgt_vocab.items()}

# ─── Instantiate encoder & decoder with correct hyperparams ─
# From your best run (cap3ahet):
#   emb_dim=64, hid_dim=256, enc_layers=1, dec_layers=3, cell=LSTM, dropout=0.2, use_attn=True
enc = Encoder(
    inp_dim = len(src_vocab),
    emb_dim = 64,
    hid_dim = 256,
    n_layers= 1,
    cell    = 'LSTM',
    dropout = 0.2
)
dec = Decoder(
    out_dim = len(tgt_vocab),
    emb_dim = 64,
    hid_dim = 256,
    n_layers= 3,
    cell    = 'LSTM',
    dropout = 0.2,
    use_attn= True
)
model = Seq2Seq(enc, dec, pad_idx, device).to(device)

# ─── Load checkpoint ─────────────────────────────────────────
state = torch.load(CHKPT, map_location=device)
model.load_state_dict(state)
model.eval()

# ─── Prepare test‐set loader (batch_size=1) ─────────────────
test_ds = TransliterationDataset(TEST_PATH, src_vocab, tgt_vocab)
test_dl = DataLoader(test_ds, batch_size=1, collate_fn=collate_fn)

# ─── Helper: get attention matrix for one example ──────────
def get_attention_matrix(src, tgt):
    """
    src: [1, S], tgt: [1, T]. Returns a NumPy array of shape [T-1, S],
    giving attention weights at each decoder step (excluding <sos>).
    """
    with torch.no_grad():
        enc_out, enc_h = model.enc(src.to(device))
        dec_h = model._init_decoder_hidden(enc_h)
        inp   = tgt[:,0].to(device)  # <sos>
        attn_list = []
        for _ in range(1, tgt.size(1)):
            logits, dec_h, attw = model.dec(inp, dec_h, enc_out)
            attn_list.append(attw[0].cpu())  # shape [S]
            inp = logits.argmax(1)
    return torch.stack(attn_list, dim=0).numpy()  # [T-1, S]

# ─── Plot “connectivity” for the first N examples ──────────
N = 3
fig, axes = plt.subplots(N, 1, figsize=(10, 3*N), constrained_layout=True)

for i, (src, tgt) in enumerate(test_dl):
    if i >= N: break

    # 1) Extract attention matrix
    attn_mat = get_attention_matrix(src, tgt)  # [T_out, S_in]
    T_out, S_in = attn_mat.shape

    # 2) Decode the raw input & target character sequences
    src_ids = src[0].cpu().tolist()
    src_chars = [inv_src[c] for c in src_ids if c != pad_idx]

    tgt_ids = tgt[0].cpu().tolist()[1:]  # skip <sos>
    tgt_chars = [inv_tgt[c] for c in tgt_ids if c not in (pad_idx, sos_idx, eos_idx)]

    # 3) Determine the single strongest input focus per output step
    peaks = attn_mat.argmax(axis=1)  # length = T_out

    ax = axes[i]
    ax.plot(peaks, list(range(T_out)), '-o', color='C1')

    # 4) Configure axes
    ax.set_title(f"Example {i+1}: “{''.join(src_chars)}” → “{''.join(tgt_chars)}”")
    ax.set_xlabel("Input character position")
    ax.set_ylabel("Decoder time step")
    ax.set_xlim(-0.5, len(src_chars)-0.5)
    ax.set_ylim(T_out-0.5, -0.5)
    ax.set_xticks(range(len(src_chars)))
    ax.set_xticklabels(src_chars, rotation=90)
    ax.set_yticks(range(T_out))
    ax.set_yticklabels(tgt_chars)

# ─── Save final figure ──────────────────────────────────────
out_file = os.path.join(BASE_DIR, 'connectivity_plots.png')
plt.savefig(out_file, dpi=300)
print("Saved connectivity plot to", out_file)
