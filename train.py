import lightning as L
from model import LitModel
from data_loader import CIFAR10DataModule
from lightning.pytorch.loggers import WandbLogger

wandb_logger = WandbLogger(project="CIFAR10 Logging")

dm = CIFAR10DataModule()
model = LitModel(*dm.dims, dm.num_classes, hidden_size=256)
trainer = L.Trainer(
    max_epochs=5,
    accelerator="auto",
    devices=1,
    logger=wandb_logger,
)
trainer.fit(model, dm)
