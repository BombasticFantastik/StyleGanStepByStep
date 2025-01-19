import torch
import torch.optim.optimizer
from torch.utils.data import DataLoader
import tqdm
from Generator import Generator
from Discriminator import Discriminator
device='cuda'
DATASET                 = "Women clothes"
START_TRAIN_AT_IMG_SIZE = 8 #The authors start from 8x8 images instead of 4x4
LEARNING_RATE           = 1e-3
BATCH_SIZES             = [256, 128, 64, 32, 16, 8]
CHANNELS_IMG            = 3
z_dim                  = 256
W_DIM                   = 256
IN_CHANNELS             = 256
LAMBDA_GP               = 10
PROGRESSIVE_EPOCHS      = [30] * len(BATCH_SIZES)



def TrainingLoop(descriminator:Discriminator,generator:Generator,dataloader:DataLoader,step:int,aplpha:float,disc_optimizer:torch.optim.AdamW,gen_optimizer:torch.optim.AdamW):
    for batch in tqdm(dataloader):
        batch=batch.to(device)
        noise=torch.randn(batch.shape,z_dim).to(device)

        fake=generator()


        

