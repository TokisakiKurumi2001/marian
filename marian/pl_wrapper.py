import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import List
from transformers import MarianTokenizer, MarianConfig, MarianMTModel
import evaluate
import numpy as np

class LitMarianMT(pl.LightningModule):
    def __init__(self, pretrained_ck: str):
        super(LitMarianMT, self).__init__()
        config = MarianConfig.from_pretrained(pretrained_ck)
        self.model = MarianMTModel(config)
        self.tokenizer = MarianTokenizer.from_pretrained(pretrained_ck)
        self.vocab_size = self.tokenizer.vocab_size
        self.loss = nn.CrossEntropyLoss()
        self.metric = evaluate.load('sacrebleu')
        self.save_hyperparameters()

    def export_model(self, path):
        self.model.save_pretrained(path)

    def __postprocess(self, predictions, labels):
        predictions = predictions.cpu().numpy()
        labels = labels.cpu().numpy()

        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)

        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]
        return decoded_preds, decoded_labels

    def training_step(self, batch, batch_idx):
        labels = batch.pop('labels')
        labels = labels.reshape(-1).long()
        logits = self.model(**batch).logits
        logits = logits.reshape(-1, self.vocab_size)
        loss = self.loss(logits, labels)
        self.log("train/loss", loss, on_epoch=True, on_step=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch.pop('labels')
        preds = self.model.generate(batch['input_ids'], max_length=128, num_beams=4)
        decoded_preds, decoded_labels = self.__postprocess(preds, labels)
        self.metric.add_batch(predictions=decoded_preds, references=decoded_labels)

    def validation_epoch_end(self, outputs):
        results = self.metric.compute()
        self.log('valid/bleu', results['score'], on_epoch=True, on_step=False, sync_dist=True)
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-5)
        return optimizer