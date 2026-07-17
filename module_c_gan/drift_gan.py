"""
DriftGAN — early generation of synthetic attack samples after a drift.

This is the novel piece proposed by the project: instead of waiting for
enough real samples of the new attack type to arrive to retrain with
confidence, a lightweight GAN is trained with the small buffer of recent
samples available at the time of the drift and synthetic variations are
generated to reinforce the retraining of the tree from the first moment.
"""

import torch
import torch.nn as nn
import torch.optim as optim


class DriftGAN:
    def __init__(self, input_dim, noise_dim=16, epochs=30, lr=0.005):
        self.input_dim = input_dim
        self.noise_dim = noise_dim
        self.epochs = epochs

        # Very lightweight generator and discriminator to not penalize
        # execution time in the stream.
        self.generator = nn.Sequential(
            nn.Linear(noise_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
        )
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        self.opt_g = optim.Adam(self.generator.parameters(), lr=lr)
        self.opt_d = optim.Adam(self.discriminator.parameters(), lr=lr)
        self.criterion = nn.BCELoss()

    def train_and_generate(self, data_buffer, num_samples=200):
        """
        Trains the GAN with a buffer of attack samples and generates
        `num_samples` synthetic variations.
        """
        if len(data_buffer) < 5:
            return []  # Not enough representative data of the new attack

        keys = list(data_buffer[0].keys())
        tensor_data = torch.tensor([[d[k] for k in keys] for d in data_buffer], dtype=torch.float32)
        batch_size = tensor_data.size(0)

        # Fast training
        for _ in range(self.epochs):
            # Discriminator
            self.opt_d.zero_grad()
            real_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)

            outputs = self.discriminator(tensor_data)
            d_loss_real = self.criterion(outputs, real_labels)

            z = torch.randn(batch_size, self.noise_dim)
            fake_data = self.generator(z)
            outputs = self.discriminator(fake_data.detach())
            d_loss_fake = self.criterion(outputs, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            self.opt_d.step()

            # Generator
            self.opt_g.zero_grad()
            z = torch.randn(batch_size, self.noise_dim)
            fake_data = self.generator(z)
            outputs = self.discriminator(fake_data)
            g_loss = self.criterion(outputs, real_labels)

            g_loss.backward()
            self.opt_g.step()

        # Generation of synthetic samples
        z = torch.randn(num_samples, self.noise_dim)
        with torch.no_grad():
            synthetic_data = self.generator(z)

        generated = []
        for i in range(num_samples):
            sample = {k: float(synthetic_data[i, j]) for j, k in enumerate(keys)}
            generated.append(sample)

        return generated
