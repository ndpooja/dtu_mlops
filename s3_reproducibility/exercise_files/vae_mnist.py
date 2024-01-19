"""Adapted from https://github.com/Jackson-Kang/Pytorch-VAE-tutorial/blob/master/01_Variational_AutoEncoder.ipynb.

A simple implementation of Gaussian MLP Encoder and Decoder trained on MNIST
"""
import logging
import os
import datetime

import hydra
from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from model import Decoder, Encoder, Model
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.utils import save_image

log = logging.getLogger(__name__)

# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = 'cpu'

@hydra.main(version_base='2.3',config_path='config_test', config_name="config.yaml")
def main(config):
    log.info(f"configuration: \n {OmegaConf.to_yaml(config)}")
    hparams = config.experiment
    mnist_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = MNIST(hparams["dataset_path"], transform=mnist_transform, train=True, download=True)
    test_dataset = MNIST(hparams["dataset_path"], transform=mnist_transform, train=False, download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=hparams["batch_size"], shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=hparams["batch_size"], shuffle=False)

    encoder = Encoder(input_dim=hparams["x_dim"], hidden_dim=hparams["hidden_dim"], latent_dim=hparams["latent_dim"])
    decoder = Decoder(latent_dim=hparams["latent_dim"], hidden_dim=hparams["hidden_dim"], output_dim=hparams["x_dim"])

    model = Model(encoder=encoder, decoder=decoder).to(device)

    def loss_function(x, x_hat, mean, log_var):
        reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")
        kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return reproduction_loss + kld
    
    optimizer = Adam(model.parameters(), lr=hparams["lr"])

    log.info("Start training VAE...")
    model.train()
    for epoch in range(hparams["epochs"]):
        overall_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            if batch_idx % 100 == 0:
                print(batch_idx)
            x = x.view(hparams["batch_size"], hparams["x_dim"])
            x = x.to(device)

            optimizer.zero_grad()

            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)

            overall_loss += loss.item()

            loss.backward()
            optimizer.step()
        log.info(f"Epoch {epoch+1} complete!,  Average Loss: {overall_loss / (batch_idx*hparams['batch_size'])}")

    print("Finish!!")

    # save weights
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_savepath = "models/" + timestamp
    os.makedirs(model_savepath)
    torch.save(model, f"{model_savepath}/trained_model.pt")

    # Generate reconstructions
    model.eval()
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(test_loader):
            if batch_idx % 100 == 0:
                print(batch_idx)
            x = x.view(hparams["batch_size"], hparams["x_dim"])
            x = x.to(device)
            x_hat, _, _ = model(x)
            break

    save_image(x.view(hparams["batch_size"], 1, 28, 28), "orig_data.png")
    save_image(x_hat.view(hparams["batch_size"], 1, 28, 28), "reconstructions.png")

    # Generate samples
    with torch.no_grad():
        noise = torch.randn(hparams["batch_size"], hparams["latent_dim"]).to(device)
        generated_images = decoder(noise)

    save_image(generated_images.view(hparams["batch_size"], 1, 28, 28), "generated_sample.png")

if __name__ == "__main__":
    main()