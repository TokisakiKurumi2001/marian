from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from marian import LitMarianMT, MarianMTDataLoader

if __name__ == "__main__":
    wandb_logger = WandbLogger(project="proj_dummy")

    lit_marianmt = LitMarianMT('Helsinki-NLP/opus-mt-en-fr')

    # dataloader
    marianmt_dataloader = MarianMTDataLoader('Helsinki-NLP/opus-mt-en-fr', max_length=128)
    [train_dataloader, valid_dataloader] = marianmt_dataloader.get_dataloader(batch_size=64, types=["train", "valid"])

    # train model
    trainer = pl.Trainer(max_epochs=2, devices=[0], accelerator="gpu", logger=wandb_logger)#, strategy="ddp")
    trainer.fit(model=lit_marianmt, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)

    # save model & tokenizer
    # lit_marianmt.export_model('marian_model/v1')
