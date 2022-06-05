import os
import random

import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from numpy import inf
from torch import nn


def fix_seed(seed):
    # random
    random.seed(seed)
    # numpy
    np.random.seed(seed)
    # pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_state(net: nn.Module, path: os.PathLike):
    torch.save(net.state_dict(), path)


def make_path(*paths):
    if len(paths) == 1:
        path = paths[0]
        os.makedirs(path, exist_ok=True)
    else:
        path = os.path.join(*paths)
        os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def make_filename(base_filename, *suffix):
    split = os.path.splitext(os.path.basename(base_filename))
    filename = split[0]
    extension = split[1]
    filename = filename + "_{}" * len(suffix) + extension
    filename = filename.format(*suffix)
    return filename


class Average:
    def __init__(self):
        self.count = 0
        self.sum = 0
        self.average = 0
        self.last = 0
        self.progress = []

    def reset(self):
        self.count = 0
        self.sum = 0
        self.average = 0
        self.last = 0
        self.progress.clear()

    def update(self, value, count=1):
        if value == inf:
            value = 0
        self.sum += value
        self.count += count
        self.average = self.sum / self.count
        self.last = value
        self.progress.append(value)

    def plot_progress(self, title=None, x_label=None, y_label=None):
        plt.title(title)
        plt.plot(np.arange(len(self.progress)), self.progress)
        plt.xlabel(xlabel=x_label)
        plt.ylabel(ylabel=y_label)
        plt.show()

    def save_progress(self, fp, title=None, x_label=None, y_label=None):
        plt.title(title)
        plt.plot(np.arange(len(self.progress)), self.progress)
        plt.xlabel(xlabel=x_label)
        plt.ylabel(ylabel=y_label)
        plt.savefig(fp)
        plt.close()


def show_pil_image(image: Image, title=None, background=None):
    plt.figure(facecolor=background)
    plt.title(title)
    plt.axis("off")
    if image.mode == 'L':
        plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    else:
        plt.imshow(image)
    plt.show()
