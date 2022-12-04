from generator import Generator
from discriminator import Discriminator
from dataset import Dataset
from experiment import Experiment
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin


def run_experiment():
    tb_logger = TensorBoardLogger(save_dir="tensorboard")
    runner = Trainer(
        logger=tb_logger,
        callbacks=[
            LearningRateMonitor(),
        ],
        strategy=DDPPlugin(find_unused_parameters=False),
        log_every_n_steps=100,
        max_epochs=2000,
    )
    runner.fit(
        Experiment(
            generator=Generator(hidden_dims=[8, 8, 16, 16, 32, 32, 64, 64, 64], dropout_rate=0.),
            discriminator=Discriminator(hidden_dims=[16, 8, 4], dropout_rate=0.),
        ),
        datamodule=Dataset()
    )


run_experiment()
