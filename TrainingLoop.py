import torch
import torch.optim.optimizer
from torch.utils.data import DataLoader
#import tqdm.std as tqdm
from tqdm import tqdm
from Generator import Generator
from Discriminator import Discriminator
import yaml



def TrainingLoop(discriminator:Discriminator,generator:Generator,dataloader:DataLoader,disc_optimizer,gen_optimizer,disc_loss_func,gen_loss_func,option):
    step=option['steps']
    alpha=float(option['alpha'])
    device=option['device']
    z_dim=option['shapes']['z_dim']
    
    for batch in (pbar:= tqdm(dataloader)):
        

        #Обучаем дискриминатор
        discriminator.zero_grad()#?
        batch=batch.to(device)
        noise=torch.randn(batch.shape[0],z_dim).to(device)

        fake_image=generator(noise,alpha,step)
        discriminator_real_pred=discriminator(batch,alpha,step)
        discriminator_fake_pred=discriminator(fake_image,alpha,step)
        real_loss_disc=disc_loss_func(discriminator_real_pred,torch.ones(batch.shape[0],1).to(device))
        fake_loss_disc=disc_loss_func(discriminator_fake_pred,torch.zeros(batch.shape[0],1).to(device))
        discriminator_loss=real_loss_disc+fake_loss_disc
        disc_loss_item=discriminator_loss.item()
        discriminator_loss.backward()
        disc_optimizer.step()

        #Обучаем генератор
        generator.zero_grad()#?
        fake_image=generator(noise,alpha,step)
        discriminator_fake_pred=discriminator(fake_image,alpha,step)
        generator_loss=gen_loss_func(discriminator_fake_pred,torch.ones(batch.shape[0],1).to(device))
        gen_loss_item=generator_loss.item()
        generator_loss.backward()
        gen_optimizer.step()
        pbar.set_description(f"disc_loss: {disc_loss_item} | gen_loss: {gen_loss_item}")
       






        

