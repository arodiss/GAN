import torch
from torch import optim
import pytorch_lightning as pl
import torchvision
from torch.utils.tensorboard import SummaryWriter
from config import BATCH_SIZE, INPUT_DIM


class Experiment(pl.LightningModule):
    def __init__(
            self,
            generator,
            discriminator,
            learning_rate=3e-4,
            scheduler_gamma=.99,
            update_training_targets_interval=20
    ) -> None:
        super(Experiment, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.learning_rate = learning_rate
        self.scheduler_gamma = scheduler_gamma
        self.step = 0
        self.update_training_targets_interval = update_training_targets_interval
        self.should_train_generator = True
        self.should_train_discriminator = True

    def generate_samples(self, batch_size: int = BATCH_SIZE):
        input_noise = torch.rand((batch_size, INPUT_DIM))
        return self.generator(input_noise)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        fake = self.generate_samples()
        real = batch
        fake_score = self.discriminator(fake)
        real_score = self.discriminator(real)
        self.update_training_targets(real_score, fake_score)

        discriminator_loss = self.discriminator.discriminator_loss(real_score, fake_score)
        generator_loss = self.generator.generator_loss(fake, fake_score)

        self.log_dict({key: val.item() for key, val in generator_loss.items()})
        self.log_dict({key: val.item() for key, val in discriminator_loss.items()})
        self.step += 1
        loss = None
        if self.should_train_generator:
            loss = generator_loss['generator_loss'] + generator_loss['homogeneity_loss']
        if self.should_train_discriminator:
            if loss is None:
                loss = discriminator_loss['total_discriminator_loss']
            else:
                loss += discriminator_loss['total_discriminator_loss']

        return loss

    def update_training_targets(self, real_score, fake_score):
        if self.step % self.update_training_targets_interval != 0:
            return
        self.should_train_generator = torch.mean(fake_score) < torch.mean(real_score) + 0.25
        self.should_train_discriminator = torch.mean(real_score) < torch.mean(fake_score) + 0.25
        self.log_dict(
            {
                "should_train_generator": float(self.should_train_generator),
                "should_train_discriminator": float(self.should_train_discriminator),
            },
            sync_dist=True
        )

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        real_score = self.discriminator(batch)
        fake = self.generate_samples()
        fake_score = self.discriminator(fake)
        self.log_dict(
            {
                "val_real_score": torch.mean(real_score),
                "val_fake_score": torch.mean(fake_score),
            },
            sync_dist=True
        )

    def on_validation_end(self) -> None:
        self.save_sampled_images()

    def save_sampled_images(self):
        self.save_images(self.generate_samples(), "generated")

    def save_images(self, images, image_type):
        images = images[:, None, :, :]
        writer = SummaryWriter(self.logger.log_dir)
        grid = torchvision.utils.make_grid(images)
        writer.add_image(image_type, grid, self.step)
        writer.close()

    def configure_optimizers(self):
        optimizers = [
            optim.Adam(
                list(self.generator.parameters()) + list(self.discriminator.parameters()),
                lr=self.learning_rate
            )
        ]
        schedulers = [
            optim.lr_scheduler.ExponentialLR(optimizers[0], gamma=self.scheduler_gamma)
        ]
        return optimizers, schedulers
