"""Grouped (per-bin) activation."""

import torch
import torch.nn as nn


class GroupedActivation(nn.Module):
  """
  G independent copies of the learned gated activation
  (activation_fcn): out = (gamma + sigmoid(beta*x)*(1-gamma))*x,
  with per-group, per-feature gamma/beta.
  """
  def __init__(self, n_groups, dim):
    super().__init__()
    # gamma/beta init to 0 (as in activation_fcn), shape (G, 1, dim)
    # to broadcast over the B axis.
    self.gamma = nn.Parameter(torch.zeros(n_groups, 1, dim))
    self.beta  = nn.Parameter(torch.zeros(n_groups, 1, dim))

  def forward(self, x):                 # x: (G, B, dim)
    inv = torch.special.expit(self.beta * x)   # sigmoid(beta*x)
    return (self.gamma + inv * (1.0 - self.gamma)) * x
