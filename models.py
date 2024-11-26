import numpy as np

import torch
import torch.nn.functional as F
from torch import nn


class RNN_vanilla(nn.Module):
    """
    Vanilla arquitechture as used in Bi & Zhou, 2019.
    """
    def __init__(self, n_inputs, hidden_size, n_outputs):
        super(RNN_vanilla, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = n_outputs
        self.i2h = nn.Linear(n_inputs + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, n_outputs)

        self.initialize_weights()

    def initialize_weights(self):
        """
        Function that initializes the input, recurrent, and output weights. 
        """
        # W_in & W_rec
        self.i2h.weight.data.fill_(0)
        K = torch.zeros((self.hidden_size, self.hidden_size + self.n_inputs))
        K[:, 0:2] = -np.sqrt(2)*torch.rand(self.hidden_size, 2) + 1/np.sqrt(2)
        K[:, 2:] = torch.normal(0., 0.3 / np.sqrt(self.hidden_size),
                                size = (self.hidden_size, self.hidden_size)) #rec no diag
        for i in range(self.hidden_size):                               #rec diag
          K[i, i+2] = 0.99
        self.i2h.weight.data += K
        # W_out
        self.h2o.weight.data.normal_(0., 0.4 / np.sqrt(self.hidden_size))
        # bias
        self.i2h.bias.data.fill_(0.)
        self.h2o.bias.data.fill_(0.)

    def forward(self, input_tensor, ends, b):
        """
        Foward pass of the network

        Parameters
        ----------
        input_tensor : time series of the inputs. Shape:
            (batch_size, time, n_inputs)
        ends : array with the time index where each trial ends. Shape:
            (1, batch_size)
        b : In case of not paralelizing the batch (current implementation),
        the current element in the batch

        Returns
        -------
        output : Time series of the network outputs for a single trial. Shape:
            (1, lenght of the trial, n_outputs)
        hidden_all : Hidden states for this trial. Shape:
            (1, lenght of the trial, hidden_size)

        """
        h0 = torch.zeros(1, self.hidden_size)
        hidden_all = torch.zeros((1, input_tensor.size(1), self.hidden_size))
        hidden = h0
        output = torch.zeros((1, int(ends[0,b]), self.output_size))

        for t in range(int(ends[0,b])):
          combined = torch.cat((input_tensor[b, t, :].view(-1, 1),
                                hidden[0, :].view(-1, 1)), 0)
          noise = (np.sqrt(2) * 0.05) * torch.randn_like(hidden)
          hidden = self.i2h(combined.reshape(-1, 1).t()) + noise
          hidden = F.softplus(hidden)

          hidden_all[0, t, :] = hidden
          output[0, t, :] = torch.clamp(self.h2o(hidden), min=-1000, max=1000)

        return output, hidden_all

    def foward_euler(self, input_tensor, ends, b, alpha):
        """
        Used in the analysis, this foward pass follows the network 
        discretization when alpha is not equal to 1.

        Parameters
        ----------
        input_tensor : time series of the inputs. Shape:
            (batch_size, time, n_inputs)
        ends : array with the time index where each trial ends. Shape:
            (1, batch_size)
        b : In case of not paralelizing the batch (current implementation),
        the current element in the batch
        alpha : alpha to be used instead of 1.

        Returns
        -------
        output : Time series of the network outputs for a single trial. Shape:
            (1, lenght of the trial, n_outputs)
        hidden_all : Hidden states for this trial. Shape:
            (1, lenght of the trial, hidden_size)

        """
        h0 = torch.zeros(1, self.hidden_size)
        hidden_all = torch.zeros((1, input_tensor.size(1), self.hidden_size))
        hidden = h0
        F_hidden = F.softplus(hidden)
        output = torch.zeros((1, int(ends[0,b]), self.output_size))

        for t in range(int(ends[0,b])):
          #Recurrent
          combined = torch.cat((input_tensor[b, t, :].view(-1, 1),
                                F_hidden[0, :].view(-1, 1)), 0)
          noise = (np.sqrt(2) * 0.05) * torch.randn_like(hidden)
          I = self.i2h(combined.reshape(-1, 1).t()) + noise

          hidden = (1 - alpha) * hidden + alpha * I
          F_hidden = F.softplus(hidden)
          hidden_all[0, t, :] = F_hidden

          #Output
          output[0, t, :] = torch.clamp(self.h2o(F_hidden),
                                        min=-1000, max=1000)

        return output, hidden_all


class RNN_Alpha(nn.Module):
    """
    Alpha arquitechture as used in Discroll et al., 2024.
    """
    def __init__(self, n_inputs, hidden_size, n_outputs,
                 alpha, diag, sigma_rec):
        super(RNN_Alpha, self).__init__()
        self.device = "cpu"
        self.hidden_size = hidden_size
        self.output_size = n_outputs
        self.diag = diag
        self.sigma_rec = sigma_rec
        self.input_size = n_inputs
        self.alpha = alpha

        # Recurrent
        self.W_rec = nn.Parameter(diag * torch.eye(hidden_size))
        self.W_in = nn.Parameter(torch.randn(hidden_size, n_inputs) /
                                 np.sqrt(n_inputs))
        self.b = nn.Parameter(torch.zeros(hidden_size))

        # Output
        self.W_out = nn.Parameter(torch.normal(0., 1/np.sqrt(hidden_size),
                                size=(n_outputs, hidden_size)))
        self.b_out = nn.Parameter(torch.zeros(n_outputs))

    def forward(self, input_tensor, ends, b):
        """
        Foward pass of the network

        Parameters
        ----------
        input_tensor : time series of the inputs. Shape:
            (batch_size, time, n_inputs)
        ends : array with the time index where each trial ends. Shape:
            (1, batch_size)
        b : In case of not paralelizing the batch (current implementation),
        the current element in the batch

        Returns
        -------
        output : Time series of the network outputs for a single trial. Shape:
            (1, lenght of the trial, n_outputs)
        hidden_all : Hidden states for this trial. Shape:
            (1, lenght of the trial, hidden_size)

        """
        h0 = torch.zeros(1, self.hidden_size).to(self.device)
        hidden_all = torch.zeros((1, input_tensor.size(1), 
                                  self.hidden_size)).to(self.device)
        hidden = h0
        output = torch.zeros((1, int(ends[0,b]), 
                              self.output_size)).to(self.device)

        for t in range(int(ends[0,b])):
          # Recurrent
          u_t = input_tensor[b, t, :]
          input_term = torch.matmul(self.W_in, u_t) \
                      .reshape(1, self.hidden_size).to(self.device)

          rec_term = torch.matmul(self.W_rec, hidden.t())\
                      .reshape(1, self.hidden_size).to(self.device)

          noise = (np.sqrt(2 / self.alpha) * self.sigma_rec) * \
                      torch.randn_like(hidden) \
                      .reshape(1, self.hidden_size).to(self.device)

          I = F.softplus(rec_term + input_term + self.b + noise)
          hidden = (1 - self.alpha) * hidden + self.alpha * I
          hidden_all[0, t, :] = hidden

          # Output
          output_term = torch.matmul(self.W_out, hidden.t()).t() + self.b_out
          output[0, t, :] = torch.clamp(output_term, min=-1000, max=1000)

        return output, hidden_all

    def simulate(self, initial_state, M):
        """
        Function used for the warm-up. Simulates the dynamics starting from 
        the initial states for M more steps.

        Parameters
        ----------
        initial_state : array with the initial states to be used. Shape:
            (n_hidden_states, hidden_size)
        M : Number of steps of the forward.

        Returns
        -------
        hidden : Final hidden states. Shape:
            (n_hidden_states, hidden_size)

        """
        hidden = initial_state

        for t in range(M):
          # Recurrent
          u_t = torch.randn(self.input_size).reshape(-1, 1)
          input_term = torch.matmul(self.W_in, u_t) \
                      .reshape(1, self.hidden_size).to(self.device)

          rec_term = torch.matmul(self.W_rec, hidden.t())\
                      .reshape(1, self.hidden_size).to(self.device)

          I = F.softplus(rec_term + input_term + self.b)
          hidden = (1 - self.alpha) * hidden + self.alpha * I

        return hidden
    
class RNN_LowRank(nn.Module):
    """
    Low rank arquitechture as used in Beiran et al., 2023.
    """
    def __init__(self, n_inputs, hidden_size, rank, n_outputs, alpha, diag):
        super(RNN_LowRank, self).__init__()
        self.device = "cpu"
        self.hidden_size = hidden_size
        self.rank = rank
        self.diag = diag
        self.output_size = n_outputs
        self.input_size = n_inputs
        self.alpha = alpha

        # Recurrent
        self.M, self.N = self.initialize_MN()
        self.W_in = nn.Parameter(torch.randn(hidden_size, n_inputs))

        # Output
        self.W_out = nn.Parameter(torch.randn(n_outputs, hidden_size))

    def initialize_MN(self):
        """
        Initialices randomly the matrices M and N so that NM is the diag times
        the Identity matrix.
        """
        cov = np.array([[1, self.diag],
                        [self.diag, 1]])
        M = np.zeros((self.hidden_size, self.rank))
        N = np.zeros((self.rank, self.hidden_size))
        
        for r in range(self.rank):
          MN_r = np.random.multivariate_normal(np.zeros(2),
                                            cov, size = self.hidden_size)
          M[:, r] = MN_r[:, 0]
          N[r, :] = MN_r[:, 1]
        
        return nn.Parameter(torch.tensor(M)), nn.Parameter(torch.tensor(N))

    def forward(self, input_tensor, ends, b):
        """
        Foward pass of the network

        Parameters
        ----------
        input_tensor : time series of the inputs. Shape:
            (batch_size, time, n_inputs)
        ends : array with the time index where each trial ends. Shape:
            (1, batch_size)
        b : In case of not paralelizing the batch (current implementation),
        the current element in the batch

        Returns
        -------
        output : Time series of the network outputs for a single trial. Shape:
            (1, lenght of the trial, n_outputs)
        hidden_all : Hidden states for this trial. Shape:
            (1, lenght of the trial, hidden_size)

        """
        h0 = torch.zeros(1, self.hidden_size).to(self.device)
        hidden_all = torch.zeros((1, input_tensor.size(1),
                                  self.hidden_size)).to(self.device)
        hidden = h0
        output = torch.zeros((1, int(ends[0,b]), 
                              self.output_size)).to(self.device)

        for t in range(int(ends[0,b])):
          # Recurrent
          u_t = input_tensor[b, t, :]
          input_term = torch.matmul(self.W_in, u_t) \
                            .reshape(1, self.hidden_size).to(self.device)

          W_rec = torch.matmul(self.N, self.M) / self.hidden_size
          rec_term = torch.matmul(W_rec, hidden.t()) \
                            .reshape(1, self.hidden_size).to(self.device)

          noise = (np.sqrt(2) * 0.05) * torch.randn_like(hidden) \
                            .reshape(1, self.hidden_size).to(self.device)

          I = rec_term + input_term + noise
          hidden = (1 - self.alpha) * hidden + self.alpha * I
          hidden = F.softplus(hidden)
          hidden_all[0, t, :] = hidden

          # Output
          output_term = torch.matmul(self.W_out, hidden.t()).t()
          output[0, t, :] = torch.clamp(output_term, min=-1000, max=1000)

        return output, hidden_all