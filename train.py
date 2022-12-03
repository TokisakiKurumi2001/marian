from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from marian import LitMarianOT, MarianOTDataLoader

if __name__ == "__main__":
    wandb_logger = WandbLogger(project="proj_dummy")

    lit_marianot = LitMarianOT('Helsinki-NLP/opus-mt-en-fr')

    # dataloader
    marianot_dataloader = MarianOTDataLoader('Helsinki-NLP/opus-mt-en-fr', max_length=128)
    [train_dataloader, valid_dataloader] = marianot_dataloader.get_dataloader(batch_size=32, types=["train", "valid"])

    # train model
    trainer = pl.Trainer(max_epochs=2, devices=[1], accelerator="gpu", logger=wandb_logger, val_check_interval=1000)#, strategy="ddp")
    trainer.fit(model=lit_marianot, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)

    # save model & tokenizer
    # lit_marianmt.export_model('marian_model/v1')
