import torch


def white(x):
    #  Wrapper to print white text in terminal
    return '\033[30m' + str(x) + '\033[0m'


def blue(x):
    #  Wrapper to print blue text in terminal
    return '\033[94m' + str(x) + '\033[0m'


def green(x):
    #  Wrapper to print blue text in terminal
    return '\033[92m' + str(x) + '\033[0m'


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    return model
