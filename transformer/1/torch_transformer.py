# coding: utf-8
from torch import nn
import torch
transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12, batch_first=False)

# batch_size是32，10是seq_len，512是embedsize
src = torch.rand((10, 32, 512))
tgt = torch.rand((20, 32, 512))

# [20, 32, 512]
out = transformer_model(src, tgt)

print(out.shape)