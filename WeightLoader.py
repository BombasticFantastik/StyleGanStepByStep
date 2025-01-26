import torch
from torch.nn import Module

def load_weights(gen_model:Module,disc_model:Module,option:dict):
    gen_dict=torch.load(f'Gen_model_weights/gen_weights{option['weights_count']}.pth',weights_only=True)
    disc_dict=torch.load(f'Disc_model_weights/disc_weights{option['weights_count']}.pth',weights_only=True)

    gen_model.load_state_dict(gen_dict)
    disc_model.load_state_dict(disc_dict)