import PIL.Image
import torch
import os
from torchvision import transforms
import PIL

#Получаем путь к файлам, получаем изображения. При выводе меняем их размер и превращаем в тензоры
class AnimeDataset(torch.utils.data.Dataset):
    def __init__(self,path):
        super(AnimeDataset,self).__init__()
        self.images=[os.path.join(path,img) for img in os.listdir(path)]
        self.transforms=transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor()
        ])
    def __len__(self):
        return len(self.images)
    def __getitem__(self, index):
        return self.transforms(PIL.Image.open(self.images[index]))
        
