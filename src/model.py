import torch.nn as nn
import transformers
import config

class BERTBaseUncased(nn.Module):
    def __init__(self):
        super(BERTBaseUncased, self).__init__()
        self.bert_path = config.BERT_PATH
        self.bert = config.BERT_MODEL
        self.bert_drop = nn.Dropout(0.25)
        self.out = nn.Linear(768, 30)

    def forward(self, ids, mask, token_type_ids):
        _, o2 = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        # print(type(o2))
        # print(o2)
        bo = self.bert_drop(o2)
        logits = self.out(bo)
        return logits