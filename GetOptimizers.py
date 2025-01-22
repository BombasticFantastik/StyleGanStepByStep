from torch.optim import Adam,AdamW
import torch
def GetGenOptimizer(gen_model,options:dict):
    optimizer_name=options['optimizers']['generator_optimizer']['name']
    lr=float(options['optimizers']['generator_optimizer']['lr'])
    match(optimizer_name):
        case 'AdamW':
            return  AdamW([{"params": [param for name, param in gen_model.named_parameters() if "map" not in name]},{"params": gen_model.mapnet.parameters(), "lr": 1e-5}],lr=lr,betas=(0.0, 0.99))
        case 'Adam':
            return Adam([{"params": [param for name, param in gen_model.named_parameters() if "map" not in name]},{"params": gen_model.mapnet.parameters(), "lr": 1e-5}],lr=lr,betas=(0.0, 0.99))
        case _:
            raise Exception("Нет такого оптимизатора")
        


def GetDiscOptimizer(disk_model,options:dict):
    optimizer_name=options['optimizers']['discriminator_optimizer']['name']
    lr=float(options['optimizers']['discriminator_optimizer']['lr'])
    match(optimizer_name):
        case 'AdamW':
            return  AdamW(disk_model.parameters(),lr=lr,betas=(0.0, 0.99))
        case 'Adam':
            return Adam(lr=lr,betas=(0.0, 0.99))
        case _:
            raise Exception("Нет такого оптимизатора")
        