from datasets import load_dataset
from torch.utils.data import DataLoader
from collections.abc import Mapping
import torch
from torch import Tensor
from typing import List, Tuple
from transformers import MarianTokenizer

class MarianOTDataLoader:
    def __init__(self, pretrained_ck: str, max_length: int):
        dataset = load_dataset('kde4', lang1='en', lang2='fr').remove_columns('id')
        dataset = dataset['train'].train_test_split(train_size=0.9, seed=42)
        dataset['valid'] = dataset.pop('test')
        self.dataset = dataset
        self.tokenizer = MarianTokenizer.from_pretrained(pretrained_ck)
        self.max_length = max_length

    def __shift_tokens_right(self, input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
        """
        Shift input ids one token to the right.
        """
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = decoder_start_token_id

        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids

    def __collate_fn(self, examples):
        if isinstance(examples, (list, tuple)) and isinstance(examples[0], Mapping):
            encoded_inputs = {key: [example[key] for example in examples] for key in examples[0].keys()}
        else:
            encoded_inputs = examples

        tok = self.tokenizer(
            encoded_inputs['en'],
            text_target=encoded_inputs['fr'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt')
        _tok = self.tokenizer(
            encoded_inputs['en'],
            text_target=encoded_inputs['en'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt')
        tok['tsl_labels'] = tok.pop('labels')
        tok['tsl_decoder_input_ids'] = self.__shift_tokens_right(tok['tsl_labels'], 59513, 59513)
        tok['tgt_labels'] = _tok.pop('labels')
        tok['tgt_decoder_input_ids'] = self.__shift_tokens_right(tok['tgt_labels'], 59513, 59513)
        
        return tok

    def get_dataloader(self, batch_size:int=16, types: List[str] = ["train", "valid"]):
        res = []
        for type in types:
            res.append(
                DataLoader(self.dataset[type]['translation'], batch_size=batch_size, collate_fn=self.__collate_fn, num_workers=24)
            )
        return res
