import torch
from Discriminator import Discriminator
from Generator import Generator
import yaml
from GetOptimizers import GetGenOptimizer,GetDiscOptimizer

option_path='config.yml'
with open(option_path,'r') as file_option:
    option=yaml.safe_load(file_option)

#Создаём генератор и его оптимизатор
anime_generator=Generator(option)
gen_optimizer=GetGenOptimizer(anime_generator,option)

#Создаём дискриминатор и его оптимизатор
anime_discriminator=Discriminator(option)
disc_optimizer=GetDiscOptimizer(anime_discriminator,option)