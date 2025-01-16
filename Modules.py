from torch import nn

class Mapping_linear_lay(nn):
    def __init__(self,z_dim,w_dim):
        super(Mapping_linear_lay,self).__init__()
        self.fc=nn.Linear(z_dim,w_dim)
        self.relu=nn.ReLU(0.2)
    def forward(self,x):
        return self.relu(self.fc(x))

class Mapping_network(nn):
    def __init__(self,z_dim,w_dim):
        super(Mapping_network,self).__init__()
        self.map0=Mapping_linear_lay(z_dim,w_dim)
        self.map1=Mapping_linear_lay(w_dim,w_dim)
        self.map2=Mapping_linear_lay(w_dim,w_dim)
        self.map3=Mapping_linear_lay(w_dim,w_dim)
        self.map4=Mapping_linear_lay(w_dim,w_dim)
        self.map5=Mapping_linear_lay(w_dim,w_dim)
        self.map6=Mapping_linear_lay(w_dim,w_dim)
        self.map7=Mapping_linear_lay(w_dim,w_dim)
    def forward(self,x):
        #не будем нормализовать, тк мы работает с нормальным стандартным распределением 
        x0=self.map0(x)
        x1=self.map1(x0)
        x2=self.map2(x1)
        x3=self.map3(x2)
        x4=self.map4(x3)
        x5=self.map5(x4)
        x6=self.map6(x5)
        x7=self.map7(x6)
        return x7



class AdaIN(nn):
    def __init__(self,w_dim,x_dim):
        super(AdaIN,self).__init__()
        self.instance_norm=nn.InstanceNorm2d(x_dim)#ЭТО ШУМ, ЗДЕСЬ МЫ РАБОТАЕМ С ШУМОМ, А НЕ С ВЕКТОРОМ СТИЛЯ
        self.style_scale=Mapping_linear_lay(w_dim,x_dim)#ЗДЕСЬ МЫ РАБОТАЕМ С ВЕКТОРОМ СТИЛЯ 
        self.style_bias=Mapping_linear_lay(w_dim,x_dim)
    def forward(self,x,w):
        x=self.instance_norm(x)
        style_scale=self.style_scale(w).unsqueeze(2).unsqueeze(3)#РАЗОБРАТЬСЯ С УНСКВИЗ
        style_bias=self.style_scale(w).unsqueeze(2).unsqueeze(3)
        return x*style_scale+ style_bias
