import torch
from torch.nn import Module
def save_weights(gen_model:Module,disc_model:Module,option:dict):
    torch.save(gen_model.state_dict(),f'Gen_model_weights/gen_weights{option['weights_count']}.pth')
    torch.save(disc_model.state_dict(),f'Disc_model_weights/disc_weights{option["weights_count"]}.pth')
    

def change_weights_count(option:dict):
    real_data="this is tarabarshina"
    filename=option['paths']['yml_path']
    with open(filename,'r') as file:
        data=file.readlines()
        for i in range(len(data)):
            if 'weights_count' in data[i]:
                data[i]=data[i][0:14]+' '+str(int(data[i][14:].strip())+1)
                real_data=''.join(data)
                break
            
    with open(filename,'w') as file:
        file.write(real_data)

                
    

    


