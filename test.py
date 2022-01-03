import torch
from torch.utils.data import DataLoader
import numpy as np

from dataloader import SeqReverseDS, \
                       pos_encoding_for_outputs, \
                       pos_encoding_for_inputs, \
                       embed_inputs, \
                       embed_outputs, \
                       shift_right, \
                       normalize_inputs_and_outputs, \
                       add_BOS_EOS


def test_SeqReverseDS():
  seq = SeqReverseDS(3, 10, 2, seed=0)

  assert len(seq) == 10
  for inp, out in seq:
    assert len(inp) == 3
    assert np.all(inp >= 0)
    assert np.all(inp < 2)
    assert list(inp) == list(out[::-1])

def test_pos_encoding_for_inputs():
  r = pos_encoding_for_inputs(3, 2*(3+2-1))
  expected = torch.tensor([[-0.5000, -0.2500,  0.0000,  0.2500,  0.5000]] * 2 * (3+2-1))
  assert r.allclose(expected)

def test_pos_encoding_for_outputs():
  # print("test_pos_encoding_for_outputs")
  r = pos_encoding_for_outputs(3, 2*(3+2-1))
  expected = torch.tensor(
      [
        [-0.5000,  0.5000,  0.2500,  0.0000, -0.2500],
        [-0.5000,  0.5000,  0.2500,  0.0000, -0.2500],
        [-0.2500, -0.5000,  0.5000,  0.2500,  0.0000],
        [-0.2500, -0.5000,  0.5000,  0.2500,  0.0000],
        [ 0.0000, -0.2500, -0.5000,  0.5000,  0.2500],
        [ 0.0000, -0.2500, -0.5000,  0.5000,  0.2500],
        [ 0.2500,  0.0000, -0.2500, -0.5000,  0.5000],
        [ 0.2500,  0.0000, -0.2500, -0.5000,  0.5000]
       ])
  assert  r.allclose(expected)

def test_embed_inputs():
  len_sent = 2
  vocab_size = 4
  x = torch.tensor([[3, 0],
                   [2, 3]])
  y = torch.tensor([[0, 3],
                [3, 2]])
  x_s, y_s, masks_s = shift_right(x, y, vocab_size)
  x_s, y_s = normalize_inputs_and_outputs(x_s, y_s, vocab_size=vocab_size)
  assert x_s.allclose(torch.tensor(
      [
       [0.0, 0.8, 0.2, 1.0],
       [0.0, 0.6, 0.8, 1.0],
       [0.0, 0.8, 0.2, 1.0],
       [0.0, 0.6, 0.8, 1.0],
       [0.0, 0.8, 0.2, 1.0],
       [0.0, 0.6, 0.8, 1.0]
       ]))
  assert y_s.allclose(torch.tensor(
      [
        [0.0, 1.0, 0.8, 0.2],
        [0.0, 1.0, 0.6, 0.8],
        [0.2, 0.0, 1.0, 0.8],
        [0.8, 0.0, 1.0, 0.6],
        [0.8, 0.2, 0.0, 1.0],
        [0.6, 0.8, 0.0, 1.0]
      ]))
  assert masks_s.allclose(torch.tensor(
      [
        [1., 0., 0., 0.],
        [1., 0., 0., 0.],
        [1., 1., 0., 0.],
        [1., 1., 0., 0.],
        [1., 1., 1., 0.],
        [1., 1., 1., 0.]
      ]))
  x_e = embed_inputs(x_s, len_sent)
  """
  for len_sent=2 there are 4 positions (including BOS and EOS).
  we map it to [-0.5, 0.5]. The positions are -0.5, -1/6, 1/6 and 0.5 (equidistant). 
  """
  assert x_e.allclose(torch.tensor(
      [[[ 0.0000, -0.5],
         [ 0.8000, -1/6],
         [ 0.2000,  1/6],
         [ 1.0000,  0.5]],

        [[ 0.0000, -0.5000],
         [ 0.6000, -1/6],
         [ 0.8000,  1/6],
         [ 1.0000,  0.5000]],

        [[ 0.0000, -0.5000],
         [ 0.8000, -1/6],
         [ 0.2000,  1/6],
         [ 1.0000,  0.5000]],

        [[ 0.0000, -0.5000],
         [ 0.6000, -1/6],
         [ 0.8000,  1/6],
         [ 1.0000,  0.5000]],

        [[ 0.0000, -0.5000],
         [ 0.8000, -1/6],
         [ 0.2000,  1/6],
         [ 1.0000,  0.5000]],

        [[ 0.0000, -0.5000],
         [ 0.6000, -1/6],
         [ 0.8000,  1/6],
         [ 1.0000,  0.5000]]]))
  y_e = embed_outputs(y_s, len_sent)
  assert y_e[:, :, 0].allclose(torch.tensor(
      [
        [0.0, 1.0, 0.8, 0.2],
        [0.0, 1.0, 0.6, 0.8],
        [0.2, 0.0, 1.0, 0.8],
        [0.8, 0.0, 1.0, 0.6],
        [0.8, 0.2, 0.0, 1.0],
        [0.6, 0.8, 0.0, 1.0]
      ]))
  assert y_e[:, :, 1].allclose(torch.tensor(
      [
        [-0.5,  0.5,  1/6, -1/6],
        [-0.5,  0.5,  1/6, -1/6],
        [-1/6, -0.5,  0.5,  1/6],
        [-1/6, -0.5,  0.5,  1/6],
        [ 1/6, -1/6, -0.5,  0.5],
        [ 1/6, -1/6, -0.5,  0.5]
      ]))



def test_add_BOS_EOS():
  res = add_BOS_EOS(torch.tensor([[0, 1, 0, 2]]), vocab_size=3)
  """
  0 is BOS
  3. is eos 
  """
  assert np.allclose(res, [[0, 1, 2, 1, 3, 4]])

def test_shift_right():
  bs = 2
  len_sent = 3
  ds_tr = SeqReverseDS(len_sent, 15, 5, seed=1)
  dl_tr = DataLoader(ds_tr, batch_size=bs, num_workers=num_workers)
  x, y = next(iter(dl_tr))

  x_s, y_s, masks_s = shift_right(x, y, vocab_size)

  assert x_s.shape == torch.Size([(len_sent+2-1) * bs, len_sent + 2])

num_workers = 0
vocab_size = 10

test_shift_right()
test_add_BOS_EOS()
test_SeqReverseDS()
test_embed_inputs()
test_pos_encoding_for_outputs()
test_pos_encoding_for_inputs()
