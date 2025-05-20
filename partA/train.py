# train.py
import wandb, torch, torch.optim as optim, torch.nn as nn
from torch.utils.data import DataLoader

# custom defined
from data_utils import TransliterationDataset, collate_fn, src_vocab, tgt_vocab
from model import Encoder, Decoder, Seq2Seq
from beam_search import beam_decode


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pad_idx = tgt_vocab['<pad>']


sweep_config = {
  'method': 'bayes',
  'metric': {'name':'val_loss','goal':'minimize'},
  'parameters':{
    'emb_dim':   {'values':[16,32,64,256]},
    'hid_dim':   {'values':[16,32,64,256]},
    'enc_layers':{'values':[1,2,3]},
    'dec_layers':{'values':[1,2,3]},
    'cell_type': {'values':['RNN','GRU','LSTM']},
    'dropout':   {'values':[0.2,0.3]},
    'beam_size': {'values':[1,3,5]},
    'lr':        {'value':1e-3},
    'batch_size':{'value':128},
      'use_attn': {'values':[True, False]}
  }
}

def calculate_accuracy(output, target, pad_idx):
    # output: [B, T, V], target: [B, T]
    with torch.no_grad():
        pred_tokens = output.argmax(dim=2)           # [B, T]
        mask        = target != pad_idx              # ignore pads
        correct     = (pred_tokens == target) & mask
        return correct.sum().float() / mask.sum().float()
    
def train_epoch(model, loader, opt, crit, pad_idx):
    model.train()
    total_loss = 0
    total_acc  = 0
    for src, tgt in loader:
        src, tgt = src.to(device), tgt.to(device)
        opt.zero_grad()
        out  = model(src, tgt, teacher_forcing=0.5)
        loss = crit(out[:,1:].reshape(-1,out.size(-1)), tgt[:,1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        opt.step()

        total_loss += loss.item()
        total_acc  += calculate_accuracy(out[:,1:], tgt[:,1:], pad_idx)
    return total_loss/len(loader), total_acc/len(loader)


def eval_epoch(model, loader, crit, beam_k, pad_idx):
    model.eval()
    total_loss = 0
    total_acc  = 0
    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            out  = model(src, tgt, teacher_forcing=0.0)
            loss = crit(out[:,1:].reshape(-1,out.size(-1)), tgt[:,1:].reshape(-1))

            total_loss += loss.item()
            total_acc  += calculate_accuracy(out[:,1:], tgt[:,1:], pad_idx)
    return total_loss/len(loader), total_acc/len(loader)

def sweep_run():
    wandb.init()
    cfg = wandb.config
    
    # data
    ds = TransliterationDataset('/kaggle/input/transliteration-9123/hi.translit.sampled.train.tsv', src_vocab, tgt_vocab)
    dv = TransliterationDataset('/kaggle/input/transliteration-9123/hi.translit.sampled.dev.tsv', src_vocab, tgt_vocab)
    dl = DataLoader(ds, batch_size=cfg.batch_size, collate_fn=collate_fn, shuffle=True)
    val_dl = DataLoader(dv, batch_size=cfg.batch_size, collate_fn=collate_fn)
    
    # model
    enc = Encoder(len(src_vocab), cfg.emb_dim, cfg.hid_dim, cfg.enc_layers, cfg.cell_type, cfg.dropout)
    dec = Decoder(len(tgt_vocab), cfg.emb_dim, cfg.hid_dim, cfg.dec_layers, cfg.cell_type, cfg.dropout, cfg.use_attn)
    model = Seq2Seq(enc,dec,pad_idx,device).to(device)
    opt   = optim.Adam(model.parameters(), lr=cfg.lr)
    crit  = nn.CrossEntropyLoss(ignore_index=pad_idx)
    
    for epoch in range(1, 11):
        tr_loss, tr_acc = train_epoch(model, dl, opt, crit, pad_idx)
        vl_loss, vl_acc = eval_epoch(model, val_dl, crit, cfg.beam_size, pad_idx)
        wandb.log({
            'epoch':       epoch,
            'train_loss':  tr_loss,
            'train_acc':   tr_acc,
            'val_loss':    vl_loss,
            'val_acc':     vl_acc
        })
            
    # save best
    torch.save(model.state_dict(),'model_use_attn.pt')
    wandb.save('model_use_attn.pt')
    
if __name__=='__main__':
    sweep_id = wandb.sweep(sweep_config, project='dakshina-translit')
    wandb.agent(sweep_id, function=sweep_run, count=200)
    
