import torch
from torch import nn
from typing import List
from torch.nn import functional as F
from config import IMAGE_HEIGHT, IMAGE_WIDTH, BATCH_SIZE, INPUT_DIM


class Generator(nn.Module):
    def __init__(
        self,
        hidden_dims: List[int],
        dropout_rate: float
    ) -> None:
        super(Generator, self).__init__()
        previous_dim = INPUT_DIM
        modules = []
        for dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(
                        previous_dim,
                        dim,
                    ),
                    nn.LeakyReLU(),
                    nn.Dropout(dropout_rate)
                )
            )
            previous_dim = dim
        modules.append(
            nn.Sequential(
                nn.Linear(
                    hidden_dims[-1],
                    IMAGE_HEIGHT * IMAGE_WIDTH,
                ),
                nn.Sigmoid(),
            ),
        )
        self.model = nn.Sequential(*modules)

    def forward(self, input_noise):
        result = self.model(input_noise)
        return torch.reshape(result, (-1, IMAGE_HEIGHT, IMAGE_WIDTH))

    @staticmethod
    def generator_loss(fake_images, fake_image_scores):
        homogeneity_loss = 0
        for first_index in range(BATCH_SIZE):
            for second_index in range(first_index+1, BATCH_SIZE-first_index):
                homogeneity_loss = homogeneity_loss - torch.sum(torch.abs(fake_images[first_index]-fake_images[second_index]))
        generator_loss = F.binary_cross_entropy(fake_image_scores, torch.ones_like(fake_image_scores))
        return {
            'homogeneity_loss': homogeneity_loss/1000,
            'generator_loss': generator_loss,
        }

