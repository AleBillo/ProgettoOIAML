import torch.optim as optim
import torch.nn as nn

def get_optimizer(model, optimizer_config):
    name = optimizer_config.get("name", "Adam")
    lr = optimizer_config.get("lr", 0.001)
    if name == "Adam":
        return optim.Adam(model.parameters(), lr=lr)
    elif name == "SGD":
        momentum = optimizer_config.get("momentum", 0.9)
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    else:
        raise ValueError(f"Unknown optimizer: {name}")

def get_loss_function(loss_name):
    if loss_name == "CrossEntropyLoss":
        return nn.CrossEntropyLoss()
    elif loss_name == "MSELoss":
        return nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")

def get_scheduler(optimizer, scheduler_config):
    name = scheduler_config.get("name", "StepLR")
    if name == "StepLR":
        step_size = scheduler_config.get("step_size", 10)
        gamma = scheduler_config.get("gamma", 0.5)
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif name == "ReduceLROnPlateau":
        factor = scheduler_config.get("factor", 0.1)
        patience = scheduler_config.get("patience", 5)
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=factor, patience=patience)
    else:
        raise ValueError(f"Unknown scheduler: {name}")
