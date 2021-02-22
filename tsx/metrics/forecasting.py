import torch

# TODO: Make pytorch-independent
def smape(a, b):
    nom = torch.abs(a - b)
    denom = torch.abs(a) + torch.abs(b)
    return torch.mean(nom / denom)