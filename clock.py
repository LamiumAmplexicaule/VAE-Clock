import os
import re
import tkinter as tk
from argparse import ArgumentParser
from datetime import datetime

import torch
from PIL import ImageTk, Image
from PIL.Image import Resampling
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose
from torchvision.utils import make_grid

import models


class Application(tk.Frame):
    def __init__(self, master, model_path, hidden_features, z_dims, image_size=64, fs=1000):
        super().__init__(master)
        self.master.title("VAE Clock")
        self.image_size = image_size
        self.fs = fs
        self.canvas = tk.Canvas(self.master, width=self.image_size * 6, height=self.image_size)
        self.canvas.pack(expand=True)
        self.time_image = None

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = models.VAE(28 * 28, hidden_features, z_dims).to(device)
        model.load_state_dict(torch.load(model_path))
        transform = Compose(
            [
                ToTensor(),
            ]
        )
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)
        model.eval()
        numbers = {i: [] for i in range(10)}
        with torch.no_grad():
            for i, (input_data, label) in enumerate(test_loader):
                input_data = input_data.to(device)
                output, mean, var = model(input_data)
                z = model.sampling(mean, var)
                for j in range(len(label)):
                    numbers[label.cpu()[j].item()].append(z[j])
        z_mean = {}
        for i in range(10):
            num_z = torch.vstack(numbers[i])
            mean = torch.mean(num_z, 0)
            z_mean[i] = mean
        self.images = [
            Image.fromarray(
                make_grid(
                    model.decoder(
                        torch.lerp(
                            z_mean[(int(i / self.fs)) % 10],
                            z_mean[(int(i / self.fs) + 1) % 10],
                            (i % self.fs) / self.fs
                        )
                    ).view(1, 28, 28)).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            ).resize((self.image_size, self.image_size), resample=Resampling.BICUBIC) for i in range(10 * self.fs)
        ]
        self.time()

    def time(self):
        now = datetime.now()
        show_time = now.strftime("%H:%M:%S")
        hour_t0 = (now.hour % 10) / 10
        hour_t1 = now.minute / 60
        minute_t0 = (now.minute % 10) / 10
        minute_t1 = now.second / 60
        second_t0 = (now.second % 10) / 10
        second_t1 = now.microsecond / 10 ** 6
        time_image = Image.new("RGB", (self.image_size * 6, self.image_size), "white")

        hour, minute, second = show_time.split(":")
        # hour
        time_image.paste(self.images[int((hour_t0 + int(hour[0])) * self.fs)], (0, 0))
        time_image.paste(self.images[int((hour_t1 + int(hour[1])) * self.fs)], (self.image_size, 0))
        # minutes
        time_image.paste(self.images[int((minute_t0 + int(minute[0])) * self.fs)], (2 * self.image_size, 0))
        time_image.paste(self.images[int((minute_t1 + int(minute[1])) * self.fs)], (3 * self.image_size, 0))
        # seconds
        time_image.paste(self.images[int((second_t0 + int(second[0])) * self.fs)], (4 * self.image_size, 0))
        time_image.paste(self.images[int((second_t1 + int(second[1])) * self.fs)], (5 * self.image_size, 0))

        self.time_image = ImageTk.PhotoImage(time_image)
        self.canvas.create_image(0, 0, anchor='nw', image=self.time_image)
        self.canvas.after(int(1000 / self.fs), self.time)


def main():
    parser = ArgumentParser()
    parser.add_argument('model_path', type=str)
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--fs', type=int, default=1000)
    args = parser.parse_args()
    model_path = args.model_path
    model_name = os.path.basename(model_path)
    z_dims = int(re.search(r"z(\d+)", model_name).group(1))
    hidden_features = int(re.search(r"h(\d+)", model_name).group(1))
    image_size = args.image_size
    fs = args.fs
    root = tk.Tk()
    app = Application(root, model_path, z_dims=z_dims, hidden_features=hidden_features, image_size=image_size, fs=fs)
    app.mainloop()


if __name__ == '__main__':
    main()
