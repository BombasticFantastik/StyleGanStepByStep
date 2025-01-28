import torch.nn as nn
import torch
from Dataset import AnimeDataset
from Models.Modules import AdaIN,ConvLay,Mapping_network,GenBlock,InjectNoise
import torch.nn.functional as F
from torch.nn import Module
import yaml




class Generator(Module):
    def __init__(self,option:dict,img_size=3):
        super(Generator,self).__init__()

        z_dim=option['shapes']['z_dim']
        w_dim=option['shapes']['w_dim']
        input_size=option['shapes']['in_channels']
        factors=option['factors']

        self.mapnet=Mapping_network(z_dim,w_dim)

        self.const=nn.Parameter(torch.ones((1,input_size,4,4)))

        self.noisegen0=InjectNoise(input_size)
        self.noisegen1=InjectNoise(input_size)      

        self.ada0=AdaIN(w_dim,input_size)
        self.ada1=AdaIN(w_dim,input_size)
        
        self.relu=nn.LeakyReLU(0.2)
        self.inital_conv=nn.Conv2d(input_size,input_size,kernel_size=3,padding=1,stride=1)
        self.rgblay=ConvLay(input_size,img_size,kernel_size=1,stride=1,padding=0)

        self.prog_blocs=nn.ModuleList([])
        self.rgb_layers=nn.ModuleList([self.rgblay])

        for i in range(len(factors)-1):
            in_size=int(input_size*factors[i])
            out_size=int(input_size*factors[i+1])
            
            self.prog_blocs.append(GenBlock(in_size,out_size,w_dim))
            self.rgb_layers.append(ConvLay(out_size,img_size,kernel_size=1,stride=1,padding=0))


    def fade_in(self,alpha,upscales,generated):
        return torch.tanh(alpha*generated+(1-alpha)*upscales)
        
    def forward(self,noice,aplpha,steps):

        w=self.mapnet(noice)
        x=self.inital_conv(self.ada0(self.noisegen0(self.const),w))
        x=self.ada1(self.relu(self.noisegen1(x)),w)

        if steps==0:
            return self.rgblay(x)
        

        for step in range(steps):
            
            upscaled=F.interpolate(x,scale_factor=2,mode='bilinear')
            
            x=self.prog_blocs[step](upscaled,w)
            
        final_out=self.rgb_layers[steps-1](upscaled)
        final_ups=self.rgb_layers[steps](x)

        return self.fade_in(aplpha,final_ups,final_out)
        


        
        