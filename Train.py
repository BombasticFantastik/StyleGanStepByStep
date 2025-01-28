import torch
from Dataset import AnimeDataset
from Models.Discriminator import Discriminator
#from Modules import ConvLay
import gc
from Models.Generator import Generator
import yaml
from Utils.GetOptimizers import GetGenOptimizer,GetDiscOptimizer
from Loop.TrainingLoop import TrainingLoop
from torch.nn import BCELoss

option_path='Config.yml'
with open(option_path,'r') as file_option:
    option=yaml.safe_load(file_option)

device=option['device']

# #Создаём датасет
anime_dataset=AnimeDataset(option)


# #Создаём даталодер
anime_dataloader=torch.utils.data.DataLoader(anime_dataset,batch_size=16,shuffle=True,drop_last=True)#.to(device)

# #Создаём генератор и его оптимизатор
anime_generator=Generator(option).to(device)
gen_optimizer=GetGenOptimizer(anime_generator,option)
gen_loss_func=BCELoss()

# #Создаём дискриминатор и его оптимизатор
anime_discriminator=Discriminator(option).to(device)
disc_optimizer=GetDiscOptimizer(anime_discriminator,option)
disc_loss_func=BCELoss()

#Очищаем память
if option['device']=='cuda':
    torch.cuda.empty_cache()
    gc.collect()

TrainingLoop(discriminator=anime_discriminator,generator=anime_generator,dataloader=anime_dataloader,disc_optimizer=disc_optimizer,gen_optimizer=gen_optimizer,disc_loss_func=disc_loss_func,gen_loss_func=gen_loss_func,option=option)