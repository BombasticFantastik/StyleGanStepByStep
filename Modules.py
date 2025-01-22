from torch.nn import Module
from torch import nn
import torch

class Mapping_linear_lay(Module):
    def __init__(self,z_dim,w_dim):
        super(Mapping_linear_lay,self).__init__()
        self.fc=nn.Linear(z_dim,w_dim)
        self.relu=nn.ReLU(0.2)
    def forward(self,x):
        return self.relu(self.fc(x))
        #return 1

class Mapping_network(Module):
    def __init__(self,z_dim,w_dim):
        super(Mapping_network,self).__init__()
        self.map0=Mapping_linear_lay(z_dim,w_dim)
        self.map1=Mapping_linear_lay(w_dim,w_dim)
        self.map2=Mapping_linear_lay(w_dim,w_dim)
        self.map3=Mapping_linear_lay(w_dim,w_dim)
        self.map4=Mapping_linear_lay(w_dim,w_dim)
        self.map5=Mapping_linear_lay(w_dim,w_dim)
        self.map6=Mapping_linear_lay(w_dim,w_dim)
        self.map7=Mapping_linear_lay(w_dim,w_dim)
    def forward(self,x):
        #не будем нормализовать, тк мы работает с нормальным стандартным распределением 
        x0=self.map0(x)
        x1=self.map1(x0)
        x2=self.map2(x1)
        x3=self.map3(x2)
        x4=self.map4(x3)
        x5=self.map5(x4)
        x6=self.map6(x5)
        x7=self.map7(x6)
        return x7



class AdaIN(Module):
    def __init__(self,w_dim,x_dim):
        super(AdaIN,self).__init__()
        self.instance_norm=nn.InstanceNorm2d(x_dim)#ЭТО ШУМ, ЗДЕСЬ МЫ РАБОТАЕМ С ШУМОМ, А НЕ С ВЕКТОРОМ СТИЛЯ
        self.style_scale=Mapping_linear_lay(w_dim,x_dim)#ЗДЕСЬ МЫ РАБОТАЕМ С ВЕКТОРОМ СТИЛЯ 
        self.style_bias=Mapping_linear_lay(w_dim,x_dim)
    def forward(self,x,w):
        x=self.instance_norm(x)
        style_scale=self.style_scale(w).unsqueeze(2).unsqueeze(3)#РАЗОБРАТЬСЯ С УНСКВИЗ
        style_bias=self.style_scale(w).unsqueeze(2).unsqueeze(3)
        return x*style_scale+ style_bias

class InjectNoise(Module):
    def __init__(self,x_dim):
        super(InjectNoise,self).__init__()
        self.weight = nn.Parameter(torch.zeros((1,x_dim,1,1)))
    def forward(self,x):
        noise = torch.randn((x.shape[0],1,x.shape[2],x.shape[3]),device=x.device)
        return x+self.weight*noise

class ConvLay(Module):
    def __init__(self,input_size,output_size,kernel_size=3,stride=1,padding=1):
        super(ConvLay,self).__init__()
        #self.conv=nn.Conv2d(input_size=input_size,output_size=output_size,kernel_size,stride,padding)
        #output_size=int(output_size)
        
        self.conv=nn.Conv2d(in_channels=input_size,out_channels=output_size,kernel_size=kernel_size,stride=stride,padding=padding)
    def forward(self,x):
        conv_x=self.conv(x)
        return conv_x
    
class GenBlock(Module):
    def __init__(self,input_size,output_size,w_dim):
        super(GenBlock,self).__init__()

        self.conv0=ConvLay(input_size,output_size)  
        self.conv1=ConvLay(output_size,output_size)     

        self.noise0=InjectNoise(output_size)
        self.noise1=InjectNoise(output_size)

        self.Ada0=AdaIN(w_dim=w_dim,x_dim=output_size)
        self.Ada1=AdaIN(w_dim=w_dim,x_dim=output_size)

        self.relu=nn.LeakyReLU(0.2)

    def forward(self,x,w):
        x0= self.Ada0(self.relu(self.noise0(self.conv0(x))),w)
        x1= self.Ada1(self.relu(self.noise1(self.conv1(x0))),w)
        return x1
