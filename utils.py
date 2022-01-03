import torch.nn.functional as fn

from dataclasses import dataclass

@dataclass
class EncDims:
  """Class to store dims of vars"""
  len_sent: int # length of the sentance
  dim_enc: int # dimentionality of encoding for each position in the sentance

  @property
  def size(self):
    return self.len_sent * self.dim_enc

def loss_fn(output, target, len_sent):
  return fn.cross_entropy(output, target) 

def calc_min_max_cross_entropy(num_classes):
  assert num_classes > 0
  max_log_sm = np.log(np.exp(1) / (np.exp(1) + (num_classes-1) * np.exp(0)))
  min_log_sm = np.log(np.exp(0) / (np.exp(1) + (num_classes-1) * np.exp(0)))
  t = [1.] + [0,]*(num_classes-1)
  t = torch.tensor(t)

  print(fn.log_softmax(t))
  print(min_log_sm, max_log_sm)
