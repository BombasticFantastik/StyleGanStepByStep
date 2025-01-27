import torch
from Generator import Generator
import torch
import matplotlib.pyplot as plt
import yaml
from PIL import Image
import os
import numpy as np

option_path='Config.yml'
with open(option_path,'r') as file_option:
    option=yaml.safe_load(file_option)


def save_images(gen_model:Generator,option:dict):
    noise=torch.randn(option['shapes']['batch_size'],option['shapes']['z_dim'])
    created_images=gen_model(noise,float(option['alpha']),6)
    created_images=created_images.detach().numpy().transpose(1,2,0)
    img=Image.fromarray((created_images*255).astype(np.uint8)).convert('RGB')
    img.save(f"Generated_images/anime_face{len(os.listdir('Generated_images'))}.jpeg")

def make_images(gen_model:Generator,option:dict):
    noise=torch.randn(option['shapes']['batch_size'],option['shapes']['z_dim'])
    created_images=gen_model(noise,float(option['alpha']),6)
    #print(created_images.shape)
    return created_images[0].detach().numpy().transpose(1,2,0)
    #plt.imshow(created_images[0].detach.numpy().transpose(1,2,0))
anime_generator=Generator(option).to(option['device'])
gen_dict=torch.load(f'Gen_model_weights/gen_weights{option['weights_count']}.pth',weights_only=True)
anime_generator.load_state_dict(gen_dict)
numpy_img=make_images(anime_generator,option)
#print(numpy_img.max(),numpy_img.min())
img=Image.fromarray((numpy_img*255).astype(np.uint8)).convert('RGB')
img.save(f"Generated_images/anime_face{len(os.listdir('Generated_images'))}.jpeg")


    
    