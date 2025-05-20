# generate_predictions.py

import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from data_utils import TransliterationDataset, collate_fn, src_vocab, tgt_vocab
from model import Encoder, Decoder, Seq2Seq

# ─────────────── Paths ───────────────────
BASE_DIR = 'partA'
TEST_PATH = os.path.join(BASE_DIR, 'lexicons', 'hi.translit.sampled.test.tsv')
CHKPT_ATTN    = os.path.join(BASE_DIR, 'model_use_attn.pt')
CHKPT_NO_ATTN = os.path.join(BASE_DIR, 'model_without_attn.pt')

# ─────────────── Device & PAD ────────────
device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pad_idx = tgt_vocab['<pad>']

# ─────────── Infer Hyperparameters ────────
def infer_hparams(state_dict):
    """From a loaded state_dict, infer emb_dim, hid_dim, layers, cell, use_attn."""
    # embedding size
    emb_dim = state_dict['enc.emb.weight'].shape[1]
    # hidden size and gate‐count → cell type
    hh = state_dict['enc.rnn.weight_hh_l0']  # [num_gates * hid_dim, hid_dim]
    hid_dim = hh.shape[1]
    num_gates = hh.shape[0] // hid_dim
    cell_type = {1: 'RNN', 3: 'GRU', 4: 'LSTM'}[num_gates]
    # number of layers = count of weight_ih_l* keys
    enc_layers = len([k for k in state_dict if k.startswith('enc.rnn.weight_ih_l')])
    dec_layers = len([k for k in state_dict if k.startswith('dec.rnn.weight_ih_l')])
    # detect attention: decoder input size = emb_dim + hid_dim if attn
    # we check any dec.rnn.weight_ih_l0 second dim == emb_dim+hid_dim
    w_ih = state_dict['dec.rnn.weight_ih_l0']
    input_size = w_ih.shape[1]
    use_attn = (input_size == emb_dim + hid_dim)
    return emb_dim, hid_dim, enc_layers, dec_layers, cell_type, use_attn

# ────────────── Load Model ────────────────
def load_model(path):
    """Load a Seq2Seq from checkpoint, inferring its exact config."""
    state = torch.load(path, map_location=device)
    e_dim, h_dim, e_layers, d_layers, c_type, u_attn = infer_hparams(state)

    enc = Encoder(len(src_vocab), e_dim, h_dim, e_layers, c_type, dropout=0.0)
    dec = Decoder(len(tgt_vocab), e_dim, h_dim, d_layers, c_type, dropout=0.0,
                  use_attn=u_attn)
    model = Seq2Seq(enc, dec, pad_idx, device).to(device)
    model.load_state_dict(state)   # exact match guaranteed
    model.eval()
    return model, u_attn

# ─────────── Prepare Test Set ────────────
test_ds = TransliterationDataset(TEST_PATH, src_vocab, tgt_vocab)
test_dl = DataLoader(test_ds, batch_size=128, collate_fn=collate_fn)

# ─────────── Load Both Models ────────────
model_attn,    uses_attn_1 = load_model(CHKPT_ATTN)
model_no_attn, uses_attn_2 = load_model(CHKPT_NO_ATTN)

print(f"ATTN model uses_attn={uses_attn_1}")
print(f"NO_ATTN model uses_attn={uses_attn_2}")

# ─────────── Greedy Decode + Save ─────────
def generate_preds(model, loader, out_file, collect_attn=False):
    os.makedirs(os.path.dirname(out_file) or '.', exist_ok=True)
    all_preds = []
    attn_mats = []

    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            B, T     = tgt.size()

            # encode
            enc_out, enc_h = model.enc(src)
            dec_h = model._init_decoder_hidden(enc_h)
            inp   = tgt[:,0]  # <sos>
            batch_attn = []

            # decode steps
            preds = torch.zeros(B, T, dtype=torch.long, device=device)
            for t in range(1, T):
                out_logits, dec_h, attw = model.dec(
                    inp, dec_h,
                    enc_out if model.dec.use_attn else None
                )
                top1 = out_logits.argmax(1)
                preds[:,t] = top1
                inp = top1
                if collect_attn:
                    batch_attn.append(attw.cpu())

            # to strings + optional attn mats
            for i in range(B):
                seq = []
                for idx in preds[i].cpu().tolist():
                    if idx==tgt_vocab['<eos>']: break
                    seq.append(next(c for c,v in tgt_vocab.items() if v==idx))
                all_preds.append(''.join(seq))

                if collect_attn:
                    mat = torch.stack([step[i] for step in batch_attn], dim=0)
                    attn_mats.append(mat.numpy())

    # write to file
    with open(out_file, 'w', encoding='utf-8') as f:
        for line in all_preds:
            f.write(line + '\n')

    return attn_mats

# vanilla
_ = generate_preds(
    model_no_attn, test_dl,
    out_file='predictions_without_attention.txt',
    collect_attn=False
)

# attention
attn_maps = generate_preds(
    model_attn, test_dl,
    out_file='predictions_with_attention.txt',
    collect_attn=True
)

# ─────────── Plot 3×3 Heatmaps ───────────
os.makedirs('attention_heatmaps', exist_ok=True)
plt.figure(figsize=(10,10))

for i in range(min(9, len(attn_maps))):
    mat = attn_maps[i]
    src_str, tgt_str = test_ds.pairs[i]
    src_chars = list(src_str)+['<eos>']
    tgt_chars = list(tgt_str)+['<eos>']

    ax = plt.subplot(3,3,i+1)
    cax= ax.matshow(mat, cmap='viridis')
    plt.colorbar(cax, ax=ax)
    ax.set_xticks(range(len(src_chars)))
    ax.set_yticks(range(len(tgt_chars)))
    ax.set_xticklabels(src_chars, rotation=90)
    ax.set_yticklabels(tgt_chars)
    ax.set_xlabel('in'); ax.set_ylabel('out')

plt.tight_layout()
plt.savefig('attention_heatmaps/heatmaps_3x3.png')
print("✅ Done: predictions_*.txt written and heatmaps_3x3.png generated.")
