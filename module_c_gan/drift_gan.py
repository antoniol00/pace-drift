"""
DriftGAN — generación temprana de muestras sintéticas de ataque tras un drift.

Esta es la pieza novedosa que propone el proyecto: en lugar de esperar a que
lleguen suficientes muestras reales del nuevo tipo de ataque para reentrenar
con confianza, se entrena una GAN ligera con el pequeño buffer de muestras
recientes disponible en el momento del drift y se generan variaciones
sintéticas para reforzar el reentrenamiento del árbol desde el primer
instante.
"""

import torch
import torch.nn as nn
import torch.optim as optim


class DriftGAN:
    def __init__(self, input_dim, noise_dim=16, epochs=30, lr=0.005):
        self.input_dim = input_dim
        self.noise_dim = noise_dim
        self.epochs = epochs

        # Generador y discriminador muy ligeros para no penalizar el tiempo
        # de ejecución en el stream.
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
        Entrena la GAN con un buffer de muestras de ataque y genera
        `num_samples` variaciones sintéticas.
        """
        if len(data_buffer) < 5:
            return []  # No hay suficientes datos representativos del nuevo ataque

        keys = list(data_buffer[0].keys())
        tensor_data = torch.tensor([[d[k] for k in keys] for d in data_buffer], dtype=torch.float32)
        batch_size = tensor_data.size(0)

        # Entrenamiento rápido
        for _ in range(self.epochs):
            # Discriminador
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

            # Generador
            self.opt_g.zero_grad()
            z = torch.randn(batch_size, self.noise_dim)
            fake_data = self.generator(z)
            outputs = self.discriminator(fake_data)
            g_loss = self.criterion(outputs, real_labels)

            g_loss.backward()
            self.opt_g.step()

        # Generación de muestras sintéticas
        z = torch.randn(num_samples, self.noise_dim)
        with torch.no_grad():
            synthetic_data = self.generator(z)

        generated = []
        for i in range(num_samples):
            sample = {k: float(synthetic_data[i, j]) for j, k in enumerate(keys)}
            generated.append(sample)

        return generated
