import numpy as np
import matplotlib.pyplot as plt
import torch

class TICT():
    def __init__(self, alpha):
        self.alpha = alpha
        self.tau = 0.025 # seconds
        self.alpha =  0.2 # 1
        self.step = self.alpha * self.tau
        
        self.t_max_start = 0.5
        self.stim1s = np.array([0.05, 0.1, 0.15, 0.2, 0.25,
                                0.35, 0.4, 0.45, 0.5, 0.55]) #Short Range
        self.mid_T = 0.3
        self.t_max_stim1 = self.stim1s[-1]
        self.t_max_delay = 0.5 #2.
        self.t_response = 0.3
        self.n_trials = 10000 #Precalculate trails
        self.n_outputs = 2
        self.n_inputs = 2
        self.n_help_t = 6
        self.t_final = self.t_max_start + self.t_max_stim1 + \
                        self.t_max_delay + self.t_response
        self.sigma_input = 0.01
    
    def time2indx(self, t):
        return np.uint32(np.rint(t / self.step))
    
    def create_trials(self):
      self.tv = np.arange(0, self.t_final, self.step)
      self.lentv = len(self.tv)
      self.trials = np.zeros((self.lentv, self.n_trials))
      self.correct_responses = np.zeros((self.n_outputs, self.lentv, self.n_trials))
      self.go_cue = np.zeros((self.lentv, self.n_trials))
      self.trial_times = np.zeros((self.n_help_t, self.n_trials), dtype="uint32")
      self.objetives = np.zeros((1, self.n_trials))
    
      for e in range(self.n_trials):
        self.trials[:, e] = 0 + np.sqrt(2) * self.sigma_input * \
            np.random.normal(0, 1, self.lentv)
        self.go_cue[:, e] = 0 + np.sqrt(2) * self.sigma_input * \
            np.random.normal(0, 1, self.lentv)
    
        T1 = np.random.choice(np.arange(2, self.time2indx(self.t_max_start))) #start
        T2 = self.time2indx(np.random.choice(self.stim1s))             #stim1
        T3 = self.time2indx(self.t_max_delay)                          #delay
        T4 = 0                                                         #stim2
        T5 = self.time2indx(self.t_response)                           #resp
    
        self.trial_times[0, e] = T1
        self.trial_times[1, e] = T2
        self.trial_times[2, e] = T3
        self.trial_times[3, e] = T1 + T2 + T3 + T4 + T5  #task end
        self.trial_times[4, e] = T1 + T2 + T3            #after delay
        self.trial_times[5, e] = T4
    
        self.trials[T1:T1+T2, e] += 1
        self.go_cue[T1+T2+T3:T1+T2+T3+T5, e] += 1
        if T2 > self.time2indx(self.mid_T):
          self.correct_responses[0, T1+T2+T3+T4:T1+T2+T3+T4+T5, e] = 1
          self.objetives[0,e] = 0
        else:
          self.correct_responses[1, T1+T2+T3+T4:T1+T2+T3+T4+T5, e] = 1
          self.objetives[0,e] = 1
    
    def plot_task_example(self):
        plt.subplot(211)
        plt.plot(self.tv, self.trials[:, 1], color="blue")
        mid_indx = self.trial_times[0, 1] + self.time2indx(self.mid_T)
        plt.vlines(self.tv[mid_indx], 0, 1, linestyle="--", color="gold")
        plt.plot(self.tv, self.correct_responses[0, :, 1], color="k")
        plt.legend(["input", "boundary", "target 1"])
        
        plt.title("Trial 1", weight="bold")
        plt.subplot(212)
        plt.plot(self.tv, self.go_cue[:, 1], color="red")
        plt.plot(self.tv, self.correct_responses[1, :, 1], color="k")
        plt.xlabel("Time [s]")
        plt.legend(["go-cue", "target 2"])
        plt.show()
        
    def get_random_trials(self, batch_size):
          inputs = np.zeros((batch_size, self.lentv, self.n_inputs))
          labels = np.zeros((batch_size, self.lentv, self.n_outputs))
          starts = np.zeros((1, batch_size), dtype="uint32")
          ends = np.zeros((1, batch_size), dtype="uint32")
        
          #for evaluation
          Ts = np.zeros((self.n_help_t, batch_size), dtype="uint32")
          obj = np.zeros((1,batch_size))
        
          for b in range(batch_size):
            r = np.random.randint(self.n_trials)
            inputs[b, :, 0] = self.trials[:, r]
            inputs[b, :, 1] = self.go_cue[:, r]
            labels[b, :, 0] = self.correct_responses[0, :, r]
            labels[b, :, 1] = self.correct_responses[1, :, r]
            starts[0, b] = self.trial_times[0, r]
            ends[0, b] = self.trial_times[3, r]
        
            #for evaluation
            Ts[:, b] = self.trial_times[:, r]
            obj[0, b] = self.objetives[0, r]
        
          return inputs, labels, starts, ends, Ts, obj
        
    def get_data(self, inputs, labels, device="cpu"):
          data = torch.from_numpy(inputs).type(torch.FloatTensor).to(device=device)
          targets = torch.from_numpy(labels).type(torch.FloatTensor).to(device=device)
          return data, targets
        
        
        
