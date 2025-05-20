# data_utils.py
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class TransliterationDataset(Dataset):
    def __init__(self, path, src_vocab, tgt_vocab, max_len=32):
        df = pd.read_csv(path, sep='\t', names=['devanagari','roman','dont_care'])
        df = df.dropna()
        self.pairs = df[['roman','devanagari']].values.tolist()
        self.src_vocab, self.tgt_vocab = src_vocab, tgt_vocab
        self.max_len = max_len

    def __len__(self): return len(self.pairs)

    def __getitem__(self, i):
        src, tgt = self.pairs[i]
        # src: [c1,c2,...] -> [..., <eos>]
        src_ids = [self.src_vocab[c] for c in src][:self.max_len] + [self.src_vocab['<eos>']]
        # tgt: [<sos>, c1,c2,..., <eos>]
        tgt_ids = [self.tgt_vocab['<sos>']] + \
                  [self.tgt_vocab[c] for c in tgt][:self.max_len] + \
                  [self.tgt_vocab['<eos>']]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)

def build_vocab(pairs, specials=['<pad>','<sos>','<eos>']):
    chars = set(''.join(pairs))
    idx = {tok:i for i,tok in enumerate(specials)}
    for c in sorted(chars):
        idx[c] = len(idx)
    return idx

def collate_fn(batch):
    srcs, tgts = zip(*batch)
    srcs_p = pad_sequence(srcs, batch_first=True, padding_value=src_vocab['<pad>'])
    tgts_p = pad_sequence(tgts, batch_first=True, padding_value=tgt_vocab['<pad>'])
    return srcs_p, tgts_p

# Build vocabs once
df = pd.read_csv('partA/lexicons/hi.translit.sampled.train.tsv', sep='\t', names=['devanagari','roman','dont_care'])
df = df.dropna()
src_vocab = build_vocab(df['roman'])
tgt_vocab = build_vocab(df['devanagari'])