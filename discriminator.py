import torch
from torch import nn
from typing import List
from torch.nn import functional as F
from config import IMAGE_HEIGHT, IMAGE_WIDTH


class Discriminator(nn.Module):
    def __init__(
        self,
        hidden_dims: List[int],
        dropout_rate: float
    ) -> None:
        super(Discriminator, self).__init__()
        previous_dim = IMAGE_WIDTH * IMAGE_HEIGHT
        modules = []
        for dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(
                        previous_dim,
                        dim,
                    ),
                    nn.LeakyReLU(),
                    nn.Dropout(dropout_rate),
                )
            )
            previous_dim = dim
        modules.append(nn.Sigmoid())
        self.model = nn.Sequential(*modules)

    def forward(self, input_image):
        return self.model(torch.reshape(input_image, (input_image.shape[0], -1)))

    @staticmethod
    def discriminator_loss(real_score, fake_score):
        real_loss = F.binary_cross_entropy(real_score, torch.ones_like(real_score))
        fake_loss = F.binary_cross_entropy(fake_score, torch.zeros_like(fake_score))
        return {
            'real_loss': real_loss,
            'fake_loss': fake_loss,
            'total_discriminator_loss': real_loss + fake_loss,
        }
