"""64 x 64 DCGAN with RGB output and a logits-based discriminator."""

import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, latent_dim=100, width=64):
        super().__init__()
        self.latent_dim = latent_dim
        # Start at 4 x 4; subsequent blocks double resolution to 64 x 64.
        layers = [nn.ConvTranspose2d(latent_dim, width * 8, 4, 1, 0, bias=False),
                  nn.BatchNorm2d(width * 8), nn.ReLU(True)]
        for incoming, outgoing in [(8, 4), (4, 2), (2, 1)]:
            layers += [nn.ConvTranspose2d(width * incoming, width * outgoing, 4, 2, 1, bias=False),
                       nn.BatchNorm2d(width * outgoing), nn.ReLU(True)]
        layers += [nn.ConvTranspose2d(width, 3, 4, 2, 1, bias=False), nn.Tanh()]
        self.network = nn.Sequential(*layers)

    def forward(self, z):
        # Turn each latent vector into a 1 x 1 feature map.
        return self.network(z.reshape(-1, self.latent_dim, 1, 1))


class Discriminator(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        # Halve spatial resolution in each block, ending with one logit per image.
        layers = [nn.Conv2d(3, width, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, True)]
        for incoming, outgoing in [(1, 2), (2, 4), (4, 8)]:
            layers += [nn.Conv2d(width * incoming, width * outgoing, 4, 2, 1, bias=False),
                       nn.BatchNorm2d(width * outgoing), nn.LeakyReLU(0.2, True)]
        # BCEWithLogitsLoss applies the sigmoid internally for numerical stability.
        layers += [nn.Conv2d(width * 8, 1, 4, 1, 0, bias=False)]
        self.network = nn.Sequential(*layers)

    def forward(self, images):
        return self.network(images).flatten()


def initialize(module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
    # BatchNorm scales start near one, unlike convolution weights near zero.
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)


def adversarial_step(generator, discriminator, real, noise, optimizer_g, optimizer_d):
    """One D update, then one non-saturating G update using the same latent batch."""
    criterion = nn.BCEWithLogitsLoss()
    discriminator.requires_grad_(True)
    optimizer_d.zero_grad(set_to_none=True)
    generated = generator(noise)
    real_logits = discriminator(real)
    # Detach fake images so the D update cannot change G.
    fake_logits = discriminator(generated.detach())
    loss_d = criterion(real_logits, torch.ones_like(real_logits))
    loss_d = loss_d + criterion(fake_logits, torch.zeros_like(fake_logits))
    loss_d.backward()
    optimizer_d.step()

    # Gradients must flow through D to G, but D's parameters are fixed in this step.
    discriminator.requires_grad_(False)
    optimizer_g.zero_grad(set_to_none=True)
    # Recompute predictions after updating D; retain the gradient path to G.
    updated_logits = discriminator(generated)
    loss_g = criterion(updated_logits, torch.ones_like(updated_logits))
    loss_g.backward()
    optimizer_g.step()
    discriminator.requires_grad_(True)
    # Return detached losses and diagnostic probabilities for logging only.
    return torch.stack([loss_d.detach(), loss_g.detach(), real_logits.sigmoid().mean().detach(),
                        fake_logits.sigmoid().mean().detach()])
