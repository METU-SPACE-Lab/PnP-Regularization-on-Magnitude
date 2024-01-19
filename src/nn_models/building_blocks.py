import torch.nn as nn

class CBNR3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding='same'):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        return self.relu(self.bn(self.conv(x)))

class CBNR3Dx2(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, bias=True):
        super().__init__()
        self.Pass1=CBNR3D(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias)
        self.Pass2=CBNR3D(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias)

    def forward(self,x):
        return self.Pass2(self.Pass1(x))

class ResBlock3D(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, bias=False):
        super().__init__()
        self.cbnr3dx2=CBNR3Dx2(in_channels=in_channels,out_channels=out_channels,
            kernel_size=kernel_size,bias=bias)
    
    def forward(self,x):
        return x+self.cbnr3dx2(x)


class ResCRC(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, bias=False,padding='same'):
        super().__init__()
        self.C1=nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)
        self.r = nn.ReLU()
        self.C2=nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding, bias=bias)

    def forward(self,x):
        return x + self.C2(self.r(self.C1(x)))



class Tconv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, bias=False) -> None:
        super().__init__()
        self.tconv=nn.ConvTranspose3d(in_channels=in_channels,out_channels=out_channels,
        kernel_size=kernel_size,stride=stride, bias=bias)

    def forward(self,x):
        return self.tconv(x)

class Sconv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, bias=False) -> None:
        super().__init__()
        self.sconv=nn.Conv3d(in_channels=in_channels,out_channels=out_channels,
        kernel_size=kernel_size,stride=stride,bias=bias)

    def forward(self,x):
        return self.sconv(x)



class AffirmModule():
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def device(net):
        print(f"checking device")
        for name, param in net.named_parameters():
            print(name,param.device)
        print(f"-------------------------------------")

    @staticmethod
    def requires_grad(net):
        print(f"checking req.grad.")
        for name, param in net.named_parameters():
            print (name, param.requires_grad)
        print(f"-------------------------------------")
    
    @staticmethod
    def training(net):
        print(f"checking training")
        for name, param in net.named_parameters():
            print (name, param.training)
        print(f"-------------------------------------")



