import torch, torch.nn.functional as F
from queue import PriorityQueue
from math import log

class BeamNode:
    def __init__(self, hidden, prev, tok, logp, length):
        self.hidden, self.prev, self.tok = hidden, prev, tok
        self.logp, self.len = logp, length
    def score(self): return self.logp / float(self.len)
    
def beam_decode(model, src, src_vocab, tgt_vocab, beam_k=3, max_len=32, device='cpu'):
    model.eval()
    with torch.no_grad():
        enc_out, hidden = model.enc(src)
        # init beam
        init_tok = torch.tensor([tgt_vocab['<sos>']], device=device)
        node = BeamNode(hidden, None, init_tok, 0.0, 1)
        pq = PriorityQueue(); pq.put((-node.score(), node))
        end_beams = []
        while not pq.empty():
            _, n = pq.get()
            if n.tok.item()==tgt_vocab['<eos>'] and n.prev is not None:
                end_beams.append((n.score(), n))
                if len(end_beams)>=beam_k: break
            inp = n.tok
            pred, hid, _ = model.dec(inp, n.hidden, enc_out if model.dec.use_attn else None)
            logps = F.log_softmax(pred, dim=1)
            topv, topi = logps.topk(beam_k)
            for i in range(beam_k):
                tok_i = topi[0][i].unsqueeze(0)
                score = n.logp + topv[0][i].item()
                new_node = BeamNode(hid, n, tok_i, score, n.len+1)
                pq.put((-new_node.score(), new_node))
        # backtrack best
        best = sorted(end_beams, key=lambda x: x[0], reverse=True)[0][1]
        seq = []
        while best.prev is not None:
            seq.append(best.tok.item()); best = best.prev
        return seq[::-1]