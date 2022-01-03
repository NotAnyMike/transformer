from tqdm import tqdm

from pytorch_nn_tools.devices import to_device
import torch
from torch.utils.data import DataLoader
import numpy as np

from transformer import Transformer
from dataloader import SeqReverseDS, shift_right, normalize_inputs_and_outputs, embedding
from utils import loss_fn, EncDims

len_sent = 15
len_sent_with_enc = len_sent+2
dim_enc = 2

num_workers = 0
batch_size = 50
num_steps = 1e5
vocab_size = 100
h = 20
n = 10
lr = 0.001

cpu = torch.device("cpu")
cuda0 = torch.device('cuda:0')
device = cpu


ds_tr = SeqReverseDS(len_sent, num_samples=1000000, vocab_size=vocab_size, seed=1)
ds_val = SeqReverseDS(len_sent, num_samples=10000, vocab_size=vocab_size, seed=2)

dl_tr = DataLoader(ds_tr, batch_size=batch_size, num_workers=num_workers)
dl_val = DataLoader(ds_val, batch_size=batch_size, num_workers=num_workers)

tr = Transformer(h, n=n, 
                 target_vocab_size=vocab_size+2, # we can predict all the words + BOS + EOS
                 enc_dims=EncDims(len_sent_with_enc, dim_enc))

tr = tr.to(device=device)

optim = torch.optim.Adam(tr.parameters(), lr=lr)


@torch.no_grad()
def validate(tr, dl_val, device):
  acc_strict = []
  acc_easy = []

  for x,y in dl_val:
    x_s, y_s, masks_s = shift_right(x, y, vocab_size)
    
    x_sn, y_sn = normalize_inputs_and_outputs(x_s, y_s, vocab_size=vocab_size)
    x_p, y_p, masks_p = embedding(x_sn, y_sn, masks_s, max_pos=len_sent)
    x_p, y_p, masks_p, y_s = to_device((x_p, y_p, masks_p, y_s), device=device)

    out = tr(x_p, y_p, masks_p)

    pred = out.argmax(dim=1)
    actual = y_s[:, -1].int()
    pairwise_match = pred == actual
    strict_match = torch.all(pairwise_match).int().item()
    ext_batch_size = x_p.shape[0]
    easy_match = pairwise_match.int().sum().item()
    acc_strict.append(strict_match)
    acc_easy.append(easy_match / ext_batch_size)
  acc_strict = np.mean(acc_strict)
  acc_easy = np.mean(acc_easy)
  
  print(f"acc_strict {acc_strict}; acc_easy={acc_easy}")

for epoch in tqdm(range(30)):
  for i, (x,y) in enumerate(dl_tr):
    x_s, y_s, masks_s = shift_right(x, y, vocab_size)
    
    x_sn, y_sn = normalize_inputs_and_outputs(x_s, y_s, vocab_size=vocab_size)
    x_p, y_p, masks_p = embedding(x_sn, y_sn, masks_s, max_pos=len_sent)
    # print(f"x_p.shape {x_p.shape}")
    # print(f"y_p.shape {y_p.shape}")
    x_p, y_p, masks_p, y_s = to_device((x_p, y_p, masks_p, y_s), device=device)

    out = tr(x_p, y_p, masks_p)

    # print(f"out: {out.shape}")
    loss = loss_fn(out, y_s[:, -1].long(), len_sent)
    optim.zero_grad()
    loss.backward()
    optim.step()
    if i % 100 == 0:
      print(i, loss)
    if i % 1000 == 0:
      validate(tr, dl_val, device)
      
