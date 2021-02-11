import timm
import torch
import torch.nn as nn

model = timm.create_model('vit_base_patch16_384', pretrained=True)

class ViT(nn.Module):
    def __init__(self, model=model):
        super(ViT, self).__init__()
        
        self.model = model
        self.model.head = nn.Linear(768, 5, bias=True)
    
    def forward(self, x):
        x = self.model(x)
        return x