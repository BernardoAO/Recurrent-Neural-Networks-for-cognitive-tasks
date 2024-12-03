import numpy as np
import torch 
import torch.nn.functional as F
import tasks
import models
import model_parameters
import os

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

def load_model(model_alias, task_name, parameters = [], new = False):
    """
    Loads a model with custom parameters or default ones if nothing is passed.
    The variable model_alias can be vanilla, alpha or low_rank.
    """
    
    if model_alias == "vanilla":
        if not parameters:
            parameters = model_parameters.vanilla_parameters
        task = load_task(task_name, parameters)
        model = models.RNN_vanilla(task.n_inputs, parameters["hidden_size"],
                                 task.n_outputs)
        
    elif model_alias == "alpha":
        if not parameters:
            parameters = model_parameters.alpha_parameters      
        task = load_task(task_name, parameters)
        model = models.RNN_Alpha(task.n_inputs, parameters["hidden_size"],
                                 task.n_outputs, task.alpha,
                                 parameters["diag"], parameters["sigma_rec"])
        
    elif model_alias == "low_rank":
        if not parameters:
            parameters = model_parameters.low_rank_parameters
        task = load_task(task_name, parameters)
        model = models.RNN_LowRank(task.n_inputs, parameters["hidden_size"],
                                 task.n_outputs, task.alpha,
                                 parameters["diag"], parameters["rank"])
        
    else:
        raise "Invalid model name"
    
    if not new:
        os.chdir("saved_models/")
        model.load_state_dict(torch.load(parameters["model_name"], 
                                         weights_only=False))
        os.chdir("..")        
        
    return model, task, parameters

def load_task(task_name, parameters):
    if task_name == "TICT":
        task = tasks.TICT(parameters["alpha"])
        task.create_trials()
    
    else:
        raise "Invalid task name"
        
    return task