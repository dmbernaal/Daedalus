#FastAI's XResnet modified to use SeLU activation function, SXResNet 
#https://github.com/fastai/fastai/blob/master/fastai/vision/models/xresnet.py
# initial modifications: mxresnet by lessw2020 - github:  https://github.com/lessw2020/mish
# appended modifications sxresnet by dmbernaal - github:  https://github.com/dmbernaal/sxresnet


# ADDED:
# SELU Activation
# selu_normal_: weight initialization: weight = weight / math.sqrt(1./fan_in)
# removed batch norm


from fastai.torch_core import *
import torch.nn as nn
import torch,math,sys
import torch.utils.model_zoo as model_zoo
from functools import partial
#from ...torch_core import Module
from fastai.torch_core import Module

import torch.nn.functional as F  #(uncomment if needed,but you likely already have it)

        
class Selu(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = torch.tensor(1.6732632423543772848170429916717)
        self.scale = torch.tensor(1.0507009873554804934193349852946)
        
    def forward(self, x):
        return self.scale * torch.where(x>=0.0, x, self.alpha * torch.exp(x) - self.alpha)
    
# MODIFIED: INIT
# w = w / math.sqrt(1/.fan_in)
def selu_normal_(tensor, mode1='fan_in', mode2='fan_out'):
    fan_in = nn.init._calculate_correct_fan(tensor, mode1)
    fan_out = nn.init._calculate_correct_fan(tensor, mode2)
    with torch.no_grad():
        return torch.randn(fan_in, fan_out) / math.sqrt(1./fan_in)
    
nn.init.selu_normal_ = selu_normal_ # adding modified init
        
# MODIFIED: Initialization
# from https://github.com/fastai/fastai/blob/5c51f9eabf76853a89a9bc5741804d2ed4407e49/fastai/layers.py
def conv1d(ni:int, no:int, ks:int=1, stride:int=1, padding:int=0, bias:bool=False):
    "Create and initialize a `nn.Conv1d` layer with spectral normalization."
    conv = nn.Conv1d(ni, no, ks, stride=stride, padding=padding, bias=bias)
    nn.init.selu_normal_(conv.weight)  # nn.init.kaiming_normal_(conv.weight)
    if bias: conv.bias.data.zero_()
    return spectral_norm(conv)



# Adapted from SelfAttention layer at https://github.com/fastai/fastai/blob/5c51f9eabf76853a89a9bc5741804d2ed4407e49/fastai/layers.py
# Inspired by https://arxiv.org/pdf/1805.08318.pdf
class SimpleSelfAttention(nn.Module):
    
    def __init__(self, n_in:int, ks=1, sym=False):#, n_out:int):
        super().__init__()
           
        self.conv = conv1d(n_in, n_in, ks, padding=ks//2, bias=False)      
       
        self.gamma = nn.Parameter(tensor([0.]))
        
        self.sym = sym
        self.n_in = n_in
        
    def forward(self,x):
        
        
        if self.sym:
            # symmetry hack by https://github.com/mgrankin
            c = self.conv.weight.view(self.n_in,self.n_in)
            c = (c + c.t())/2
            self.conv.weight = c.view(self.n_in,self.n_in,1)
                
        size = x.size()  
        x = x.view(*size[:2],-1)   # (C,N)
        
        # changed the order of mutiplication to avoid O(N^2) complexity
        # (x*xT)*(W*x) instead of (x*(xT*(W*x)))
        
        convx = self.conv(x)   # (C,C) * (C,N) = (C,N)   => O(NC^2)
        xxT = torch.bmm(x,x.permute(0,2,1).contiguous())   # (C,N) * (N,C) = (C,C)   => O(NC^2)
        
        o = torch.bmm(xxT, convx)   # (C,C) * (C,N) = (C,N)   => O(NC^2)
          
        o = self.gamma * o + x
        
          
        return o.view(*size).contiguous()        
        


    
    
__all__ = ['SXResNet', 'sxresnet18', 'sxresnet34', 'sxresnet50', 'sxresnet101', 'sxresnet152']

# or: ELU+init (a=0.54; gain=1.55)
act_fn = Selu() #nn.ReLU(inplace=True)

class Flatten(Module):
    def forward(self, x): return x.view(x.size(0), -1)
        
# MODIFIED: Will init weights: w = w / math.sqrt(1./fan_in)
def init_cnn(m):
    if getattr(m, 'bias', None) is not None: nn.init.constant_(m.bias, 0)
    if isinstance(m, (nn.Conv2d,nn.Linear)): nn.init.selu_normal_(m.weight)
    for l in m.children(): init_cnn(l)        
        

def conv(ni, nf, ks=3, stride=1, bias=False):
    return nn.Conv2d(ni, nf, kernel_size=ks, stride=stride, padding=ks//2, bias=bias)

def noop(x): return x

# MODIFIED: No batch norm
def conv_layer(ni, nf, ks=3, stride=1, act=True):
    layers = [conv(ni, nf, ks, stride=stride)]
    if act: layers.append(act_fn)
    return nn.Sequential(*layers)

class ResBlock(Module):
    def __init__(self, expansion, ni, nh, stride=1,sa=False, sym=False):
        nf,ni = nh*expansion,ni*expansion
        layers  = [conv_layer(ni, nh, 3, stride=stride),
                   conv_layer(nh, nf, 3, act=False)
        ] if expansion == 1 else [
                   conv_layer(ni, nh, 1),
                   conv_layer(nh, nh, 3, stride=stride),
                   conv_layer(nh, nf, 1, act=False)
        ]
        self.sa = SimpleSelfAttention(nf,ks=1,sym=sym) if sa else noop
        self.convs = nn.Sequential(*layers)
        # TODO: check whether act=True works better
        self.idconv = noop if ni==nf else conv_layer(ni, nf, 1, act=False)
        self.pool = noop if stride==1 else nn.AvgPool2d(2, ceil_mode=True)

    def forward(self, x): return act_fn(self.sa(self.convs(x)) + self.idconv(self.pool(x)))

def filt_sz(recep): return min(64, 2**math.floor(math.log2(recep*0.75)))

class SXResNet(nn.Sequential):
    def __init__(self, expansion, layers, c_in=3, c_out=1000, sa = False, sym= False):
        stem = []
        sizes = [c_in,32,64,64]  #modified per Grankin
        for i in range(3):
            stem.append(conv_layer(sizes[i], sizes[i+1], stride=2 if i==0 else 1))
            #nf = filt_sz(c_in*9)
            #stem.append(conv_layer(c_in, nf, stride=2 if i==1 else 1))
            #c_in = nf

        block_szs = [64//expansion,64,128,256,512]
        blocks = [self._make_layer(expansion, block_szs[i], block_szs[i+1], l, 1 if i==0 else 2, sa = sa if i in[len(layers)-4] else False, sym=sym)
                  for i,l in enumerate(layers)]
        super().__init__(
            *stem,
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            *blocks,
            nn.AdaptiveAvgPool2d(1), Flatten(),
            nn.Linear(block_szs[-1]*expansion, c_out),
        )
        init_cnn(self)

    def _make_layer(self, expansion, ni, nf, blocks, stride, sa=False, sym=False):
        return nn.Sequential(
            *[ResBlock(expansion, ni if i==0 else nf, nf, stride if i==0 else 1, sa if i in [blocks -1] else False,sym)
              for i in range(blocks)])

def sxresnet(expansion, n_layers, name, pretrained=False, **kwargs):
    model = SXResNet(expansion, n_layers, **kwargs)
    if pretrained: 
        #model.load_state_dict(model_zoo.load_url(model_urls[name]))
        print("No pretrained yet for SXResNet")
    return model

me = sys.modules[__name__]
for n,e,l in [
    [ 18 , 1, [2,2,2 ,2] ],
    [ 34 , 1, [3,4,6 ,3] ],
    [ 50 , 4, [3,4,6 ,3] ],
    [ 101, 4, [3,4,23,3] ],
    [ 152, 4, [3,8,36,3] ],
]:
    name = f'sxresnet{n}'
    setattr(me, name, partial(sxresnet, expansion=e, n_layers=l, name=name))