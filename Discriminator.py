import torch.nn as nn
from Modules import ConvLay
import torch
from torch.nn import Module




class   DiscConvBlock(Module):
    def __init__(self,input_size,output_size):
        super(DiscConvBlock,self).__init__()

        self.conv0=nn.Conv2d(input_size,output_size,kernel_size=3)#я добавил кернел сайз
        self.conv1=nn.Conv2d(output_size,output_size,kernel_size=3)#я добавил кернел сайз
        self.relu=nn.LeakyReLU(0.2,inplace=True)
    def forward(self,x):
        x=self.relu(self.conv0(x))
        x=self.relu(self.conv1(x))
        return x
    
class Discriminator(Module):
    def __init__(self,option:dict,img_size=3):
        super(Discriminator,self).__init__()

        input_size=option['shapes']['in_channels']
        factors=option['factors']

        self.relu=nn.LeakyReLU(0.2,inplace=True)

        self.prog_blocks=nn.ModuleList([])
        self.rgb_layers=nn.ModuleList([])

        for i in range(len(factors)-1,0,-1):
            in_size=int(input_size*factors[i])
            out_size=int(input_size*factors[i-1])
            self.prog_blocks.append(DiscConvBlock(in_size,out_size))
            self.rgb_layers.append(ConvLay(img_size,in_size,kernel_size=1,stride=1,padding=0))

        self.initial_rgb=ConvLay(img_size,input_size,kernel_size=1,stride=1,padding=0)
        self.rgb_layers.append(self.initial_rgb)
        self.avg_pool=nn.AvgPool2d(kernel_size=2,stride=2)

        self.final_block=nn.Sequential(
            ConvLay(input_size+1,input_size,kernel_size=3,padding=1),
            nn.LeakyReLU(0.2,inplace=True),
            ConvLay(input_size,input_size,kernel_size=4,padding=0,stride=1),
            nn.LeakyReLU(0.2,inplace=True),
            ConvLay(input_size,1,kernel_size=1,padding=0,stride=1),
            nn.Sigmoid()
        )
    def fade_in (self,alpha,downscaled,out):
        return alpha*out + (1-alpha)*downscaled

    def minibatch_std(self,x):
        batch_statistics=(
            torch.std(x,dim=0).mean().repeat(x.shape[0],1,x.shape[2],x.shape[3])
        )
        return torch.cat([x,batch_statistics],dim=1)
    def forward(self,x,alpha,steps):
        correct_step=len(self.prog_blocks)-steps
        out=self.relu(self.rgb_layers[correct_step](x))

        if steps==0:
            out=self.minibatch_std(x)
            return self.final_block(out).view(out.shape,-1)
        
        downscaled=self.relu(self.rgb_layers[correct_step+1](self.avg_pool(x)))
        out=self.avg_pool(self.prog_blocks[correct_step](out))
        out=self.fade_in(alpha,downscaled,out)

        for step in steps:
            out=self.prog_blocks[step](out)
            out=self.avg_pool(out)
        out=self.minibatch_std(out)
        return self.final_block(out).view(out.shape[0],-1)




