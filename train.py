from argparse import ArgumentParser

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor
from torchvision.utils import save_image
from tqdm import tqdm

import models
from utils import fix_seed, Average, make_path, make_filename, save_state


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    parser = ArgumentParser()
    parser.add_argument('--z_dims', default=20, type=int)
    parser.add_argument('--hidden_features', default=380, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--max_epoch', default=128, type=int)
    parser.add_argument('--seed', default=123456789, type=int)
    args = parser.parse_args()

    z_dims = args.z_dims
    hidden_features = args.hidden_features
    batch_size = args.batch_size
    max_epoch = args.max_epoch
    train_title = f'train_loss, z{z_dims}, h{hidden_features}, b{batch_size}, e{max_epoch}'
    test_title = f'test_loss, z{z_dims}, h{hidden_features}, b{batch_size}, e{max_epoch}'
    reconstruction_filename = make_filename("reconstruction.png", f'z{z_dims}', f'h{hidden_features}', f'b{batch_size}',
                                            f'{max_epoch}')
    fix_seed(args.seed)

    transform = Compose(
        [
            ToTensor(),
        ]
    )

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model = models.VAE(28 * 28, hidden_features, z_dims).to(device)
    criterion = models.VAELoss(28 * 28)
    optimizer = Adam(model.parameters(), lr=1e-3)

    train_loss_average = Average()
    test_loss_average = Average()
    with tqdm(total=max_epoch, desc="[Epoch]") as epoch_progressbar:
        for epoch in range(1, max_epoch + 1):
            epoch_train_loss_average = Average()
            epoch_test_loss_average = Average()
            with tqdm(total=len(train_loader), desc="[Train Iteration]", leave=False) as train_iteration_progressbar:
                model.train()
                for i, (input_data, _) in enumerate(train_loader):
                    input_data = input_data.to(device, non_blocking=True)
                    optimizer.zero_grad()
                    output, mean, var = model(input_data)
                    loss = criterion(output, input_data, mean, var)
                    loss.backward()
                    loss = loss.item()
                    optimizer.step()
                    train_iteration_progressbar.set_postfix_str("Train Loss: {:.5g}".format(loss))
                    train_iteration_progressbar.update()
                    epoch_train_loss_average.update(loss)

            with tqdm(total=len(test_loader), desc="[Test Iteration]", leave=False) as test_iteration_progressbar:
                model.eval()
                with torch.no_grad():
                    for i, (input_data, _) in enumerate(test_loader):
                        input_data = input_data.to(device)
                        output, mean, var = model(input_data)
                        loss = criterion(output, input_data, mean, var).item()
                        test_iteration_progressbar.set_postfix_str("Test Loss: {:.5g}".format(loss))
                        test_iteration_progressbar.update()
                        epoch_test_loss_average.update(loss)
                        if (epoch % 5 == 0 or epoch == max_epoch) and i == 0:
                            n = min(input_data.size(0), 8)
                            comparison = torch.cat([input_data[:n], output.view(args.batch_size, 1, 28, 28)[:n]])
                            save_image(comparison.cpu(), make_path('results', reconstruction_filename), nrow=n)

            epoch_progressbar.set_postfix_str("LR {:.5g}, Ave Train Loss: {:.5g}, Ave Test Loss: {:.5g}".format(
                optimizer.param_groups[0]['lr'],
                epoch_train_loss_average.average,
                epoch_test_loss_average.average
            ))
            epoch_progressbar.update()
            train_loss_average.update(epoch_train_loss_average.last)
            test_loss_average.update(epoch_test_loss_average.last)

            if epoch % 5 == 0 or epoch == max_epoch:
                with torch.no_grad():
                    sample = torch.randn(64, z_dims).to(device)
                    sample = model.decoder(sample).cpu()
                    save_image(sample.view(64, 1, 28, 28), make_path('sample', reconstruction_filename))

                save_state(model, make_path('save', f'vae_z{z_dims}_h{hidden_features}_b{batch_size}_e{epoch}.pth'))

    train_loss_average.plot_progress(title=train_title)
    train_loss_average.save_progress(
        make_path('plot', make_filename('train_loss.png', f'z{z_dims}', f'h{hidden_features}', f'b{batch_size}')),
        title=train_title
    )
    test_loss_average.plot_progress(title=test_title)
    test_loss_average.save_progress(
        make_path('plot', make_filename('test_loss.png', f'z{z_dims}', f'h{hidden_features}', f'b{batch_size}')),
        title=test_title
    )


if __name__ == '__main__':
    main()
