from marian import MarianOTDataLoader, MarianOT, MarianOTConfig
from geomloss import SamplesLoss
import torch.nn as nn
import torch

marianot_dataloader = MarianOTDataLoader('Helsinki-NLP/opus-mt-en-fr', max_length=128)
[train_dataloader] = marianot_dataloader.get_dataloader(batch_size=2, types=["train"])
for batch in train_dataloader:
    break
print(batch)
config = MarianOTConfig.from_pretrained('Helsinki-NLP/opus-mt-en-fr')
model = MarianOT(config)
tsl_labels = batch.pop('tsl_labels').reshape(-1).long()
tgt_labels = batch.pop('tgt_labels').reshape(-1).long()
res = model.forward_pass(**batch)
tgt_logits = res.tgt.logits.reshape(-1, 59514)
tsl_logits = res.tsl.logits.reshape(-1, 59514)
loss = nn.CrossEntropyLoss()

tgt_loss = loss(tgt_logits, tgt_labels)
tsl_loss = loss(tsl_logits, tsl_labels)
print(tgt_loss)
print(tsl_loss)
ot_loss_func = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
ot_loss = ot_loss_func(res.tgt_last_hidden_state, res.tsl_last_hidden_state).mean()
print(ot_loss)
print(0.05*ot_loss + 0.5*tgt_loss + 0.5*tsl_loss)
# input_ids = torch.tensor([[5118, 23501,   483, 16041,  4335,   108,  6809,     0]])
# out = model.generate(input_ids)
# print(out)
# with model.translate_self():
#     out = model.generate(input_ids)
#     print(out)