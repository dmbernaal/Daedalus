import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import math

class Selu():
    def __init__(self):
        self.alpha = torch.tensor(1.6732632423543772848170429916717)
        self.scale = torch.tensor(1.0507009873554804934193349852946)
    def __call__(self, x):
        return self.scale * torch.where(x>=0.0, x, self.alpha * torch.exp(x) - self.alpha)
    
class Linear():
    def __init__(self, w, b): self.w, self.b = w,b
    def __call__(self, x): return x@self.w + self.b
    
class AlphaDropout(nn.Module):
    # PyTorch implementation
    def __init__(self, dropout, lambd=1.0507, alpha=1.67326):
        super().__init__()
        self.lambd = lambd
        self.alpha = alpha
        self.aprime = -lambd * alpha
        
        self.q = 1 - dropout
        self.p = dropout
        
        self.a = (self.q + self.aprime**2 * self.q * self.p)**(-0.5)
        self.b = -self.a * (self.p * self.aprime)
        
    def __call__(self, x):
        if not self.training: return x # we don't want dropout at inference
        ones = torch.ones(x.size())
        mask = torch.bernoulli(ones * self.p)
        x = x.masked_fill(torch.autograd.Variable(mask.byte()), self.aprime)
        
class Model_Optimized():
    def __init__(self, data):
        self.x, self.w, self.b = data[0], data[1], data[2]
        self.linear, self.selu = Linear(self.w, self.b), Selu()
        self.layer = [self.linear, self.selu]
        
    def __call__(self, x, num_layers, plot_stats=False):
        init_stats, in_x = stats(x), x
        for i in range(num_layers):
            if not torch.isnan(x.mean()):
                for l in self.layer: x = l(x)
        final_stats = stats(x)
        
        print(f'Iteration: {i+1}')
        print(f'Initial Activation Stats: {init_stats}')
        print(f'Final Activation Stats: {final_stats}')
        
        if plot_stats:
            f, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
            sns.distplot(in_x, ax=axes[0])
            sns.distplot(x, ax=axes[1])
            
        return i+1, init_stats, final_stats
    
class Weight_Init():
    def __init__(self, fan_in, fan_out):
        self.fan_in, self.fan_out = fan_in, fan_out
        
    def randn(self): 
        return torch.randn(self.fan_in, self.fan_out)
    
    def kaiming(self): 
        return torch.randn(self.fan_in, self.fan_out) * math.sqrt(2./self.fan_in)
    
    def kaiminguni(self):
        return torch.Tensor(self.fan_in, self.fan_out).uniform_(-1,1) * math.sqrt(2./self.fan_in)
    
    def pytorch(self):
        w = torch.Tensor(self.fan_in, self.fan_out)
        return nn.init.kaiming_uniform_(w, mode='fan_in')
    
    def xavier(self): 
        return torch.Tensor(self.fan_in, self.fan_out).uniform_(-1,1) * math.sqrt(6./(self.fan_in + self.fan_out))
    
    def kaiming2(self):
        return torch.randn(self.fan_in, self.fan_out) * math.sqrt(1./self.fan_in)
    
def stats(x): return x.mean(), x.std()