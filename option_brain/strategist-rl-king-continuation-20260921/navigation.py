"""Frozen supervised known-path decoder. No optimizer or training entrypoint."""
import numpy as np
import torch
from torch import nn
class GoalHead(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(13,64),nn.Tanh(),nn.Linear(64,9))
    def forward(self,x):return self.net(x)
def features(relative_goal,next_edge,legal):
    if len(legal)!=9 or not any(legal):raise ValueError('navigation legal mask')
    return np.array([relative_goal[0]/112,relative_goal[1]/112,next_edge[0],next_edge[1],*legal],np.float32)
