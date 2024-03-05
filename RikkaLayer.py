import torch
class DenseBlock(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.I = torch.nn.Linear(dim, dim)
        self.O = torch.nn.Linear(dim, dim)
    def forward(self, x):
        x = self.I(x)
        x = torch.nn.functional.gelu(x)
        x = self.O(x)
        return x
    
class RikkaLayer(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.A = DenseBlock(dim)
        self.B = DenseBlock(dim)
        self.C = DenseBlock(dim)
        self.D = DenseBlock(dim)
        
        self.x = None
        self.dim = dim
    def reset(self, x=None):
        if (x != None):
            self.x = x
        else:
            self.x = None
    def forward(self, uI):
        out = []
        for a in range(uI.shape[1]):
            u = uI[:, a]
            if (self.x != None):
                xx = torch.nn.functional.normalize(self.x)
                uu = torch.nn.functional.normalize(u)
                self.x = self.A(xx) + self.B(uu) + self.x + u
                y =  self.C(xx) + self.D(uu)
                out.append(y.view(-1, 1, self.dim))
            else:
                uu = torch.nn.functional.normalize(u)
                self.x = self.B(uu) + u
                y = self.D(uu)
                out.append(y.view(-1, 1, self.dim))
        y = torch.cat(out, dim=1)
        return y + uI
    def step(self, u):
        if (self.x != None):
            xx = torch.nn.functional.normalize(self.x)
            uu = torch.nn.functional.normalize(u)
            self.x = self.A(xx) + self.B(uu) + self.x + u
            y =  self.C(xx) + self.D(uu)
        else:
            uu = torch.nn.functional.normalize(u)
            self.x = self.B(uu)
            y = self.D(uu)
        return y + u
