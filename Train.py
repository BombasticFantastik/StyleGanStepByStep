import torch
from Discriminator import Discriminator
from Generator import Generator
import yaml

option_path='config.yml'
with open(option_path,'r') as file_option:
    option=yaml.safe_load(file_option)

anime_generator=Generator(option['shapes'])

anime_discriminator