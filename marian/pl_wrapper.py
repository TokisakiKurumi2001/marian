import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import List
from transformers import MarianTokenizer
from marian import MarianOT, MarianOTConfig
import evaluate
import numpy as np

class LitMarianOT(pl.LightningModule):
    def __init__(self, pretrained_ck: str):
        super(LitMarianOT, self).__init__()
        config = MarianOTConfig.from_pretrained(pretrained_ck)
        self.model = MarianOT(config)
        self.tokenizer = MarianTokenizer.from_pretrained(pretrained_ck)
        self.vocab_size = self.tokenizer.vocab_size
        self.loss = nn.CrossEntropyLoss()
        self.metric_tgt = evaluate.load('sacrebleu')
        self.metric_tsl = evaluate.load('sacrebleu')
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
        tsl_labels = batch.pop('tsl_labels').reshape(-1).long()
        tgt_labels = batch.pop('tgt_labels').reshape(-1).long()

        res = self.model.forward_pass(**batch)
        tgt_logits = res.tgt.logits.reshape(-1, self.vocab_size)
        tsl_logits = res.tsl.logits.reshape(-1, self.vocab_size)
        tgt_loss = self.loss(tgt_logits, tgt_labels)
        tsl_loss = self.loss(tsl_logits, tsl_labels)
        loss = tgt_loss + tsl_loss
        self.log("train/loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        tsl_labels = batch.pop('tsl_labels')
        preds_tsl = self.model.generate(batch['input_ids'], max_length=128, num_beams=4)
        decoded_preds, decoded_labels = self.__postprocess(preds_tsl, tsl_labels)
        self.metric_tsl.add_batch(predictions=decoded_preds, references=decoded_labels)

        tgt_labels = batch.pop('tgt_labels')
        with self.model.translate_self():
            preds_tgt = self.model.generate(batch['input_ids'], max_length=128, num_beams=4)
            decoded_preds, decoded_labels = self.__postprocess(preds_tgt, tgt_labels)
            self.metric_tgt.add_batch(predictions=decoded_preds, references=decoded_labels)

    def validation_epoch_end(self, outputs):
        results = self.metric_tgt.compute()
        self.log('valid/bleu_tgt', results['score'], on_epoch=True, on_step=False, sync_dist=True)
        results = self.metric_tsl.compute()
        self.log('valid/bleu_tsl', results['score'], on_epoch=True, on_step=False, sync_dist=True)
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-5)
        return optimizer