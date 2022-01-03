import torch
import torch.nn.functional as fn
from torch import nn

from utils import EncDims
  
def attention(q: torch.Tensor, k: torch.Tensor, v:torch.Tensor, mask=None):
  """Batched mulihead attention
  
  q = k = v = torch.Tensor([nxl]) where n=sentence lenght, and l=dims of encoding
  """
  # Similarity
  x = q @ k.transpose(-2, -1)
  
  # Scale
  dk = x.shape[-2] # Changed
  x = x / dk**0.5
  
  # Mask
  if mask is not None:
    # print(f"attention: x {x.shape}, mask {mask.shape}")
    x = x * mask
    
  # Softmax
  x = fn.softmax(x, dim=-1) 
  
  return x @ v

class MultiHeadAttention(nn.Module):
  """Multi head attention
  
  h: heads
  enc_dims: dims of single sequence (nxl)
  """
  def __init__(self, h: int, enc_dims: EncDims, use_mask=False):
    super().__init__()
    self.h = h
    self.enc_dims = enc_dims # sentance encoding dimensions
    self.use_mask = use_mask
    
    # Linear layers
    self.v = nn.Linear(in_features=enc_dims.size, out_features=enc_dims.size * h)
    self.k = nn.Linear(in_features=enc_dims.size, out_features=enc_dims.size * h)
    self.q = nn.Linear(in_features=enc_dims.size, out_features=enc_dims.size * h)
    self.final = nn.Linear(in_features=enc_dims.size * h, out_features=enc_dims.size)

  def forward(self, v, k, q, mask=None):
    def apply_linear(x, linear):
      x = linear(x.reshape(-1, self.enc_dims.size))
      return x.reshape(-1, self.h, self.enc_dims.len_sent, self.enc_dims.dim_enc)

    v = apply_linear(v, self.v)
    k = apply_linear(k, self.k)
    q = apply_linear(q, self.q)
    
    if self.use_mask:
      assert mask is not None
      mask = mask.unsqueeze(1)

    # Final linear layer
    a = attention(q, k, v, mask)
    a = a.reshape(-1, self.h * self.enc_dims.size)
    result = self.final(a).reshape(-1, self.enc_dims.len_sent, self.enc_dims.dim_enc)

    return result
  
class EncoderBlock(nn.Module):
  def __init__(self, h: int, enc_dims: EncDims):
    super().__init__()
    self.h = h
    self.enc_dims = enc_dims
    
    self.linear = nn.Linear(enc_dims.size, enc_dims.size)
    self.norm1 = nn.BatchNorm1d(enc_dims.size)
    self.norm2 = nn.BatchNorm1d(enc_dims.size)
    self.mha = MultiHeadAttention(h, enc_dims=enc_dims)

    # Not in the original paper
    self.relu = nn.ReLU()
  
  def forward(self, x):
    x = self.mha(x,x,x)+x
    x = self.norm1(x.reshape(-1, self.enc_dims.size))
    x = self.linear(x)+x
    x = self.relu(x)
    x = self.norm2(x).reshape(-1, self.enc_dims.len_sent, self.enc_dims.dim_enc)
    return x
  
class DecoderBlock(nn.Module):
  def __init__(self, h, enc_dims: EncDims):
    super().__init__()
    self.h = h
    self.enc_dims = enc_dims
    
    self.linear = nn.Linear(enc_dims.size, enc_dims.size)
    self.norm1 = nn.BatchNorm1d(enc_dims.size)
    self.norm2 = nn.BatchNorm1d(enc_dims.size)
    self.norm3 = nn.BatchNorm1d(enc_dims.size)
    
    self.masked_mha = MultiHeadAttention(h, enc_dims=enc_dims, use_mask=True)
    self.mha = MultiHeadAttention(h, enc_dims=enc_dims)

    # Not in the original paper
    self.relu = nn.ReLU()

  def forward(self, inputs, outputs, masks):
    enc_dims = self.enc_dims
    x = outputs
    
    x = self.masked_mha(x, x, x, masks) + x
    x = self.norm1(x.reshape(-1, enc_dims.size)).reshape(-1, enc_dims.len_sent, enc_dims.dim_enc)

    x = self.mha(inputs, inputs, x) + x 
    x = self.norm2(x.reshape(-1, enc_dims.size))
    x = self.linear(x) + x
    x = self.relu(x)
    x = self.norm3(x)
    return x.reshape(-1, enc_dims.len_sent, enc_dims.dim_enc)
  
class Transformer(nn.Module):
  def __init__(self, h, n, target_vocab_size, enc_dims: EncDims):
    super().__init__()
    self.enc_dims = enc_dims
    self.encoders = nn.Sequential(*[EncoderBlock(h, enc_dims=enc_dims) for _ in range(n)])
    self.decoders = nn.ModuleList([DecoderBlock(h, enc_dims=enc_dims) for _ in range(n)])
    self.linear = nn.Linear(in_features=enc_dims.size, out_features=target_vocab_size)
    self.n = n

  def forward(self, inputs, outputs, masks):
    inputs = self.encoders(inputs)
    x = outputs
    for i in range(self.n):
      x = self.decoders[i](inputs, x, masks)
    x = self.linear(x.reshape(-1, self.enc_dims.size))
    # I remove softmax from here because it is automatcally computed by fn.cross_entropy. See tests below and torch docs. x = fn.softmax(x, dim=-1)
    return x

def loss_fn(output, target, len_sent):
  return fn.cross_entropy(output, target) 
