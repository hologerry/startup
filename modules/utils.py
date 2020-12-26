import torch


def tile_like(x, target):  # tile_size = 256 or 4
    x = x.view(x.size(0), x.size(1), 1, 1)
    x = x.repeat(1, 1, target.size(2), target.size(3))
    return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())  # Total parameters


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def moving_average(model, model_test, beta=0.999):
    for param, param_test in zip(model.parameters(), model_test.parameters()):
        param_test.data = torch.lerp(param.data, param_test.data, beta)
