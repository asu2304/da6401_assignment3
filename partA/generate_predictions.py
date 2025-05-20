# generate_predictions_only.py

import os
import torch
from torch.utils.data import DataLoader

from data_utils import TransliterationDataset, collate_fn, src_vocab, tgt_vocab
from model import Encoder, Decoder, Seq2Seq

# ─── Paths ───────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
TEST_PATH       = os.path.join(BASE_DIR, 'lexicons', 'hi.translit.sampled.test.tsv')
CHKPT_ATTN      = os.path.join(BASE_DIR, 'model_use_attn.pt')
CHKPT_NO_ATTN   = os.path.join(BASE_DIR, 'model_without_attn.pt')
OUT_WITH_ATTN   = 'predictions_with_attention.txt'
OUT_WITHOUT_ATTN = 'predictions_without_attention.txt'

# ─── Device & tokens ─────────────────────────────────────────
device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pad_idx = tgt_vocab['<pad>']
sos_idx = tgt_vocab['<sos>']
eos_idx = tgt_vocab['<eos>']

# ─── Infer & load attention model ─────────────────────────────
def load_model(checkpoint_path):
    # Infer hyperparameters from checkpoint
    state = torch.load(checkpoint_path, map_location='cpu')
    emb_dim    = state['enc.emb.weight'].shape[1]
    hh         = state['enc.rnn.weight_hh_l0']
    hid_dim    = hh.shape[1]
    num_gates  = hh.shape[0] // hid_dim
    cell_type  = {1:'RNN',3:'GRU',4:'LSTM'}[num_gates]
    enc_layers = len([k for k in state if k.startswith('enc.rnn.weight_ih_l')])
    dec_layers = len([k for k in state if k.startswith('dec.rnn.weight_ih_l')])
    # detect attention
    w0         = state['dec.rnn.weight_ih_l0']
    use_attn   = (w0.shape[1] == emb_dim + hid_dim)
    dropout    = 0.2  # per your best runs

    # Build model with inferred hyperparameters
    enc = Encoder(len(src_vocab), emb_dim, hid_dim, enc_layers, cell_type, dropout)
    dec = Decoder(len(tgt_vocab), emb_dim, hid_dim, dec_layers, cell_type, dropout, use_attn)
    model = Seq2Seq(enc, dec, pad_idx, device).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, use_attn

# ─── Load both models ───────────────────────────────────────
model_attn, uses_attn1    = load_model(CHKPT_ATTN)
model_no_attn, uses_attn2 = load_model(CHKPT_NO_ATTN)
print(f"Attention model uses_attn={uses_attn1}")
print(f"No-attention model uses_attn={uses_attn2}")

# ─── Prepare test DataLoader ─────────────────────────────────
test_ds = TransliterationDataset(TEST_PATH, src_vocab, tgt_vocab)
test_dl = DataLoader(test_ds, batch_size=128, shuffle=False, collate_fn=collate_fn)

# ─── Generate predictions function ─────────────────────────────
def generate_predictions(model, loader, out_file):
    all_preds = []
    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            B, T = tgt.size()
            enc_out, enc_h = model.enc(src)
            dec_h = model._init_decoder_hidden(enc_h)
            inp = tgt[:,0]  # <sos>
            preds = torch.zeros(B, T, dtype=torch.long, device=device)
            
            for t in range(1, T):
                logits, dec_h, _ = model.dec(
                    inp, dec_h,
                    enc_out if model.dec.use_attn else None
                )
                top1 = logits.argmax(1)
                preds[:,t] = top1
                inp = top1
            
            # Convert to strings
            for i in range(B):
                seq = []
                for idx in preds[i].cpu().tolist():
                    if idx == eos_idx:
                        break
                    seq.append(next(ch for ch,v in tgt_vocab.items() if v==idx))
                all_preds.append(''.join(seq))
    
    # Save to file
    with open(out_file, 'w', encoding='utf-8') as f:
        for line in all_preds:
            f.write(line + '\n')
    
    return len(all_preds)

# ─── Generate predictions for both models ───────────────────────
count_attn = generate_predictions(model_attn, test_dl, OUT_WITH_ATTN)
count_no_attn = generate_predictions(model_no_attn, test_dl, OUT_WITHOUT_ATTN)

print(f"Generated {count_attn} predictions with attention model → saved to {OUT_WITH_ATTN}")
print(f"Generated {count_no_attn} predictions without attention → saved to {OUT_WITHOUT_ATTN}")
