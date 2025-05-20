# evaluate_preds.py

import os
import sys
import pandas as pd

# ─── Paths ─────────────────────────────────────────────────────
BASE_DIR           = os.path.dirname(os.path.abspath(__file__))
TEST_TSV           = os.path.join(BASE_DIR, 'lexicons', 'hi.translit.sampled.test.tsv')
PRED_WITH_ATTN     = os.path.join(BASE_DIR, 'predictions_with_attention.txt')
PRED_WITHOUT_ATTN  = os.path.join(BASE_DIR, 'predictions_without_attention.txt')

# ─── 1) Load the test references ──────────────────────────────
if not os.path.isfile(TEST_TSV):
    print(f"Error: test TSV not found at {TEST_TSV}")
    sys.exit(1)

df = pd.read_csv(
    TEST_TSV,
    sep='\t',
    header=None,
    names=['ref','inp','_'],
    usecols=[0,1],
    dtype=str
)

# strip anything after "<eos>" if present
refs = df['ref'].str.split('<eos>').str[0].tolist()

# ─── 2) Load predictions ───────────────────────────────────────
def load_preds(path):
    if not os.path.isfile(path):
        print(f"Error: prediction file not found at {path}")
        sys.exit(1)
    with open(path, encoding='utf-8') as f:
        # strip blank lines and <eos>
        lines = [l.strip().split('<eos>')[0] for l in f if l.strip()]
    return lines

preds_attn  = load_preds(PRED_WITH_ATTN)
preds_noatt = load_preds(PRED_WITHOUT_ATTN)

# ensure lengths match
if not (len(refs)==len(preds_attn)==len(preds_noatt)):
    print("Error: Mismatch in number of examples vs. predictions")
    print(f" refs={len(refs)}, with_attn={len(preds_attn)}, without_attn={len(preds_noatt)}")
    sys.exit(1)

# ─── 3) Compute accuracies ─────────────────────────────────────
def compute_accs(preds, refs):
    total_chars   = sum(len(r) for r in refs)
    correct_chars = sum(
        1
        for p,r in zip(preds, refs)
        for pc,rc in zip(p, r)
        if pc == rc
    )
    correct_words = sum(p == r for p, r in zip(preds, refs))
    char_acc = correct_chars / total_chars * 100
    word_acc = correct_words / len(refs) * 100
    return char_acc, word_acc

char_attn, word_attn   = compute_accs(preds_attn,  refs)
char_noatt, word_noatt = compute_accs(preds_noatt, refs)

# ─── 4) Print results ─────────────────────────────────────────
print(f"With Attention   → Character‐level accuracy: {char_attn:.2f}%")
print(f"                  → Word‐level      accuracy: {word_attn:.2f}%\n")
print(f"Without Attention→ Character‐level accuracy: {char_noatt:.2f}%")
print(f"                  → Word‐level      accuracy: {word_noatt:.2f}%")
