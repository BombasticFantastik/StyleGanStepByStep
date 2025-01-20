import torch
import torch.optim.optimizer
from torch.utils.data import DataLoader
import tqdm
from Generator import Generator
from Discriminator import Discriminator
import yaml

option_path='config.yml'
with open(option_path,'r') as file_option:
    option=yaml.safe_load(file_option)
device=option['device']

def TrainingLoop(discriminator:Discriminator,generator:Generator,dataloader:DataLoader,step:int,aplpha:float,disc_optimizer:torch.optim.AdamW,gen_optimizer:torch.optim.AdamW,disc_loss_func,gen_loss_func):
    for batch in tqdm(dataloader):
        

        #Обучаем дискриминатор
        discriminator.zero_grad()#?
        batch=batch.to(device)
        noise=torch.randn(batch.shape,z_dim).to(device)
        fake_image=generator(noise,aplpha,step)
        discriminator_real_pred=discriminator(batch,aplpha,step)
        discriminator_fake_pred=discriminator(fake_image,aplpha,step)
        real_loss_disc=disc_loss_func(discriminator_real_pred,torch.ones(batch.shape[0],1))
        fake_loss_disc=disc_loss_func(discriminator_fake_pred,torch.ones(batch.shape[0],0))
        discriminator_loss=real_loss_disc+fake_loss_disc
        disc_loss_item=discriminator_loss.item()
        discriminator_loss.backward()
        disc_optimizer.step()

        #Обучаем генератор
        generator.zero_grad()#?
        fake_image=generator(noise,aplpha,step)
        discriminator_fake_pred=discriminator(fake_image,aplpha,step)
        generator_loss=gen_loss_func(discriminator_fake_pred,torch.ones(batch.shape[0],1))
        gen_loss_item=generator_loss.item()
        generator_loss.backward()
        gen_optimizer.step()






        

