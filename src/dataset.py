import config
import torch

class BertDataset:
    def __init__(self, qtitle, qbody, answer, targets):
        self.qtitle = qtitle
        self.qbody = qbody
        self.answer = answer
        self.targets = targets
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN
        self.len = len(qtitle)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        a_qtitle = str(self.qtitle[idx])
        a_qbody =  str(self.qbody[idx])
        a_answer = str(self.answer[idx])
        a_target = self.targets[idx, :]
        # print(len(a_qtitle + a_qbody))
        inputs = self.tokenizer.encode_plus(
            a_qtitle + " " + a_qbody,
            a_answer,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True
        )

        ids = inputs['input_ids']
        token_type_ids = inputs['token_type_ids']
        mask = inputs['attention_mask']

        padding_len = self.max_len - len(ids)
        padding  = [0] * padding_len
        ids = ids + padding
        token_type_ids = token_type_ids + padding
        mask =  mask + padding

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask':torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(a_target, dtype=torch.float)
        }
