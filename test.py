import os
import re
from argparse import ArgumentParser

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose
from tqdm import tqdm

import models
from utils import make_path, make_filename


def main():
    colors = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser = ArgumentParser()
    parser.add_argument('model_path', type=str)
    parser.add_argument('--batch_size', default=256, type=int)
    args = parser.parse_args()
    model_path = args.model_path
    batch_size = args.batch_size
    model_name = os.path.basename(model_path)
    z_dims = int(re.search(r"z(\d+)", model_name).group(1))
    hidden_features = int(re.search(r"h(\d+)", model_name).group(1))

    transform = Compose(
        [
            ToTensor(),
        ]
    )

    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model = models.VAE(28 * 28, hidden_features, z_dims).to(device)
    model.load_state_dict(torch.load(model_path))

    model.eval()
    zs = []
    labels = []
    with tqdm(total=len(test_loader), desc="[Test Iteration]", leave=False) as test_iteration_progressbar:
        model.eval()
        with torch.no_grad():
            for i, (input_data, label) in enumerate(test_loader):
                input_data = input_data.to(device)
                output, mean, var = model(input_data)
                z = model.sampling(mean, var)
                zs.append(z)
                labels.append(label)
                test_iteration_progressbar.update()

    labels = np.concatenate(labels)
    zs = np.concatenate(zs)
    plt.figure(figsize=(10, 10))
    points = TSNE(n_components=2, learning_rate=1.2, init='pca').fit_transform(zs)
    for p, l in zip(points, labels):
        plt.scatter(p[0], p[1], marker="${}$".format(l), c=colors[l])
    plt.savefig(make_path("plot", make_filename("t-sne.png", model_name.split()[0])))
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
