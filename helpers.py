import numpy as np
import matplotlib.pyplot as plt
import torch 
import torch.nn.functional as F


def VAA_star(hn, epsilon):
    """
    Returns the truncated VAA* as described in Lambrechts et al., 2023.
    For the set of hidden states `hn` and a tolerance of `epsilon`.
    """
    n = hn.size(0)
    vaa = torch.tensor(0.0, device=hn.device)
    tanh = torch.tanh(hn)
    epsilon_inverse = 1e-6
    
    for i in range(n):
      dist_i = torch.norm(tanh[:, :] - tanh[i, :], dim=1)
      Ci = 1.0 - F.relu(dist_i - epsilon) / (dist_i + epsilon_inverse)
      vaa = vaa + (1.0 / Ci.sum()) * (1.0 / n)
    return vaa

def check_produced(scores, T5s, b, ti):
    """
    Checks if an interval is produced. We consider the network tho produce a 
    valid response when all of the output units are under a pre treshold
    before the response time (T5s), and only one surpases the response 
    treshold at the response time.
    """
    pre_thr = 0.2
    resp_thr = 0.5
    no_early_response = torch.max(scores[b, :int(T5s[b]), :]) < pre_thr
    only_one_up = ((torch.max(scores[b, ti:ti+15, 0]) > resp_thr and
                torch.max(scores[b, ti:ti+15, 1]) < resp_thr) or
                (torch.max(scores[b, ti:ti+15, 0]) < resp_thr and
                torch.max(scores[b, ti:ti+15, 1]) > resp_thr))
    return no_early_response and only_one_up

  
