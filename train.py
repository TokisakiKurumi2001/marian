from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from compromise_marian import LitComproMar, ComproMarDataLoader

if __name__ == "__main__":
    wandb_logger = WandbLogger(project="proj_compromise_marian")

    lit_compromar = LitComproMar('Helsinki-NLP/opus-mt-en-fr', lr=3e-5)

    # dataloader
    compromar_dataloader = ComproMarDataLoader('Helsinki-NLP/opus-mt-en-fr', max_length=128)
    [train_dataloader, valid_dataloader] = compromar_dataloader.get_dataloader(batch_size=32, types=["train", "valid"])

    # train model
    trainer = pl.Trainer(max_epochs=5, devices=[1], accelerator="gpu", logger=wandb_logger, val_check_interval=1000)#, strategy="ddp")
    trainer.fit(model=lit_compromar, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)

    # save model & tokenizer
    lit_marianmt.export_model('marian_model/v1')
