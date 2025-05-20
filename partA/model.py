# model.py
import torch, torch.nn as nn, torch.nn.functional as F

class Encoder(nn.Module):
    
    def __init__(self, inp_dim, emb_dim, hid_dim, n_layers, cell, dropout):
        super().__init__()
        self.emb = nn.Embedding(inp_dim, emb_dim)
        RNN = {'RNN': nn.RNN, 'LSTM': nn.LSTM, 'GRU': nn.GRU}[cell]
        self.rnn = RNN(emb_dim, hid_dim, n_layers,
                       dropout=dropout if n_layers>1 else 0,
                       batch_first=True)
        self.drop = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [B, S]
        e = self.drop(self.emb(x))
        out, hidden = self.rnn(e)
        return out, hidden

class Attention(nn.Module):
    
    def __init__(self, hid_dim):
        super().__init__()
        self.attn = nn.Linear(hid_dim*2, hid_dim)
        self.v    = nn.Linear(hid_dim, 1, bias=False)
        
    def forward(self, hidden, enc_out):
        # hidden: [B, H], enc_out: [B, S, H]
        B, S, H = enc_out.size()
        h = hidden.unsqueeze(1).repeat(1,S,1)               # [B,S,H]
        energy = torch.tanh(self.attn(torch.cat([h,enc_out],dim=2)))  # [B,S,H]
        scores = self.v(energy).squeeze(2)                  # [B,S]
        return F.softmax(scores, dim=1)
    
class Decoder(nn.Module):
    
    def __init__(self, out_dim, emb_dim, hid_dim, n_layers, cell, dropout, use_attn=False):
        super().__init__()
        self.emb = nn.Embedding(out_dim, emb_dim)
        RNN = {'RNN': nn.RNN, 'LSTM': nn.LSTM, 'GRU': nn.GRU}[cell]
        self.rnn = RNN(emb_dim + (hid_dim if use_attn else 0),
                       hid_dim, n_layers,
                       dropout=dropout if n_layers>1 else 0,
                       batch_first=True)
        self.fc  = nn.Linear(hid_dim, out_dim)
        self.drop = nn.Dropout(dropout)
        self.use_attn = use_attn
        if use_attn: self.attn = Attention(hid_dim)
        
    def forward(self, tgt_tok, hidden, enc_out=None):
        # tgt_tok: [B], hidden: (h_n, c_n)? or h_n
        B = tgt_tok.size(0)
        t = tgt_tok.unsqueeze(1)            # [B,1]
        emb = self.drop(self.emb(t))        # [B,1,E]
        
        if self.use_attn:
            h = hidden[-1] if not isinstance(hidden, tuple) else hidden[0][-1]
            attn_w = self.attn(h, enc_out)  # [B, S]
            ctx    = torch.bmm(attn_w.unsqueeze(1), enc_out)  # [B,1,H]
            rnn_in = torch.cat([emb, ctx], dim=2)
        else:
            rnn_in = emb
        out, hidden = self.rnn(rnn_in, hidden)
        pred = self.fc(out.squeeze(1))      # [B, out_dim]
        
        return pred, hidden, (attn_w if self.use_attn else None)
    
    
class Seq2Seq(nn.Module):
    def __init__(self, enc, dec, pad_idx, device):
        super().__init__()
        self.enc, self.dec = enc, dec
        self.pad_idx = pad_idx
        self.device  = device
        
    def _init_decoder_hidden(self, enc_hidden):
        """
        Align encoder hidden states to decoder.num_layers by:
        - copying the last min(enc_layers, dec_layers) layers when
          decoder has fewer layers,
        - zero‐padding above when decoder has more layers.
        Supports both GRU/RNN (Tensor) and LSTM (tuple).
        """
        dec_layers = self.dec.rnn.num_layers

        if isinstance(enc_hidden, tuple):  # LSTM: (h, c)
            h, c = enc_hidden
            enc_layers, B, H = h.size()
            # Prepare zero states
            h0 = torch.zeros(dec_layers,  B, H, device=self.device)
            c0 = torch.zeros(dec_layers,  B, H, device=self.device)
            # Number of layers to copy
            n = min(enc_layers, dec_layers)
            # Copy last n layers from encoder into bottom of decoder state
            h0[-n:] = h[-n:]
            c0[-n:] = c[-n:]
            return (h0, c0)

        else:  # GRU or vanilla RNN
            h = enc_hidden
            enc_layers, B, H = h.size()
            h0 = torch.zeros(dec_layers, B, H, device=self.device)
            n = min(enc_layers, dec_layers)
            h0[-n:] = h[-n:]
            return h0
        
    def forward(self, src, tgt, teacher_forcing=0.5):
        B, T = tgt.size()
        out_dim = self.dec.fc.out_features
        outputs = torch.zeros(B, T, out_dim, device=self.device)

        enc_out, enc_hidden = self.enc(src)
        # Initialize decoder hidden state to match dec_layers
        dec_hidden = self._init_decoder_hidden(enc_hidden)

        inp = tgt[:,0]  # <sos>
        for t in range(1, T):
            pred, dec_hidden, _ = self.dec(
                inp, dec_hidden,
                enc_out if self.dec.use_attn else None
            )
            outputs[:,t] = pred
            top1 = pred.argmax(1)
            inp = tgt[:,t] if torch.rand(1).item() < teacher_forcing else top1
            
        return outputs