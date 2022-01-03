import numpy as np

import torch
from torch.utils.data import DataLoader

class SeqReverseDS:
  def __init__(self, len_sent, num_samples, vocab_size, seed=0):
    """
    Generates sequences of integers and reverse versions of it. 
    Each integer is from [0, vocab_size).
    The length of the sequence is `len_sent`.
    """
    self.num_samples = num_samples
    self.len_sent = len_sent
    self.vocab_size = vocab_size
    self.seed = seed

  def __len__(self):
    return self.num_samples

  def __getitem__(self, i):
    if i >= self.num_samples:
      raise StopIteration()
    rs = np.random.RandomState(seed=self.seed + i) 
    v = rs.randint(0, self.vocab_size, self.len_sent, dtype='int64')
    return v, v[::-1].copy()

  def __iter__(self):
    for i in range(len(self)):
      yield self[i]

def add_BOS_EOS(inputs, vocab_size):
  """
  x becomes x+1 for x in [0, vocab_size)
  0 is BOS
  vocab_size is EOS,
  """
  batch_size = inputs.shape[0]
  EOS = (vocab_size+1)
  inputs = torch.cat([
                      torch.zeros(batch_size, 1, device=inputs.device), 
                      inputs+1, 
                      torch.ones(batch_size, 1, device=inputs.device)* EOS], dim=-1)
  return inputs
 
def normalize_by_vocab_size(inputs, vocab_size):
  return inputs / (vocab_size + 1) # To account for 2 extra characters
 
def shift_right(inputs, outputs, vocab_size, normalize=None):
  assert normalize is None # TODO: delete after refactoring is done
  len_sent = inputs.shape[1]
  batch_size = inputs.shape[0]

  inputs = add_BOS_EOS(inputs, vocab_size)
  outputs = add_BOS_EOS(outputs, vocab_size)
  
  list_inputs = [inputs] * (len_sent+1)
  list_outputs = []
  masks = []
  for i in range(len_sent+1):
    r = torch.roll(outputs.flip(-1), i+1, 1)
    list_outputs.append(r)
    mask = [0 if j > i else 1 for j in range(len_sent+2)]
    mask = torch.Tensor([mask]*inputs.shape[0])
    masks.append(mask)
  
  list_inputs = torch.row_stack(list_inputs)
  list_outputs = torch.row_stack(list_outputs)
  masks = torch.row_stack(masks)

  return list_inputs, list_outputs, masks

def normalize_inputs_and_outputs(list_inputs, list_outputs, vocab_size):
  list_inputs = normalize_by_vocab_size(list_inputs, vocab_size)
  list_outputs = normalize_by_vocab_size(list_outputs, vocab_size)
  return list_inputs, list_outputs

def pos_encoding_for_inputs(sent_len, ext_batch_size):
  """
  ext_batch_size: batch size multiplied by number of possible shifts
  """
  pos = [list(range(0, sent_len+2))] * ext_batch_size
  pos = torch.Tensor(pos) / (sent_len + 1) 
  pos = pos - 0.5
  return pos

def embed_inputs(inputs, max_pos):
  """
  Simplistic embedding of inputs. Inputs are takens as is, positional encoding appended.
  Expects a batch with incorporated shifts.
  """
  pos = pos_encoding_for_inputs(sent_len=max_pos, ext_batch_size=inputs.shape[0])
  pos = pos.to(device=inputs.device)
  inputs = torch.cat([inputs.unsqueeze(-1), pos.unsqueeze(-1)], dim=-1)
  return inputs
  
def pos_encoding_for_outputs(max_pos, ext_batch_size):
  pos = torch.Tensor([list(range(max_pos+1, -1, -1))]) # in total max_pos + 2 positions (including BOS and EOS)

  batch_size = ext_batch_size // (max_pos+1)
  # print("real batch_size", batch_size)
  # print(pos)
  pos = torch.cat([pos] * batch_size, -2)
  # print(pos)
  pos = [torch.roll(pos, i+1, -1) for i in range(0, max_pos+1)]
  pos = torch.cat(pos, 0) / (max_pos+1)
  pos = pos - 0.5

  return pos 

def embed_outputs(outputs, max_pos):
  """
  Simplistic embedding of outputs. Outputs are takens as is, positional encoding appended.
  expects a batch with incorporated shifts
  """
  ext_batch_size = outputs.shape[0]
  pos = pos_encoding_for_outputs(max_pos=max_pos, ext_batch_size=ext_batch_size)
  pos = pos.to(device=outputs.device)
  outputs = torch.cat([outputs.unsqueeze(-1), pos.unsqueeze(-1)], dim=-1)
  return outputs

def embedding(inputs, outputs, masks, max_pos):
  inputs = embed_inputs(inputs, max_pos=max_pos)
  outputs = embed_outputs(outputs, max_pos=max_pos)
  
  masks = masks.unsqueeze(-2)
  
  return inputs, outputs, masks
