import torch
import torch.nn as nn
import numpy as np


def triu_mask(input_size, device=None):
    mask = torch.ones(input_size, input_size, device=device)
    return mask.triu(diagonal=1)

def VtoQ(V):
    Q = torch.matmul(V.T, V)
    return Q * triu_mask(Q.size(0), device=Q.device)

class QuadraticLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def init_params(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.zeros_(param)
            elif param.dim() > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.normal_(param, mean=0, std=0.01)

    def get_bhQ(self):
        raise NotImplementedError()
        
class FactorizationMachine(QuadraticLayer):
    def __init__(self, input_size, factorization_size=8, act="identity"):
        super().__init__()
        if factorization_size <= 0:
            raise ValueError("factorization_size 必須是正整數。")

        self.input_size = input_size
        self.facrotization_size = factorization_size
        self.act_name = act

        self.h = nn.Parameter(torch.empty(input_size))
        self.bias = nn.Parameter(torch.empty(1))
        self.V = nn.Parameter(torch.empty(factorization_size, input_size))
        self.init_params()

    def forward(self, x):
        linear_term = self.bias + torch.matmul(x, self.h)
        interaction_part1 = torch.matmul(x, self.V.T)
        interaction_part2 = torch.matmul(x**2, (self.V**2).T)

        interaction_term = 0.5 * torch.sum(interaction_part1**2 - interaction_part2, dim=1)
        out = linear_term + interaction_term

        if self.act_name == "sigmoid":
            return torch.sigmoid(out).view(-1, 1)
        elif self.act_name == "tanh" :
            return torch.tanh(out).view(-1, 1)
        else:
            return out.view(-1, 1)

    def get_bhQ(self, scaling=True):
        V_data = self.V.detach().cpu()
        Q = VtoQ(V_data).numpy()
        bias = self.bias.detach().cpu().item()
        h = self.h.detach().cpu().numpy()
        
        # 新增 Scaling 邏輯
        if scaling:
            h_max = np.max(np.abs(h))
            Q_max = np.max(np.abs(Q))
            scaling_factor = max(h_max, Q_max)
            if scaling_factor > 0:
                bias /= scaling_factor
                h /= scaling_factor
                Q /= scaling_factor
                
        return bias, h, Q

