from utils import loss_fn
import torch
import config
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from dataset import BertDataset
from model import BERTBaseUncased
import scipy.stats as stats
import time


def train_loop_fn(data_loader, model, optimizer, scheduler=None):
    model.train()
    device = config.DEVICE
    losses = []
    for batch_idx, data in enumerate(data_loader):
        ids = data['ids']
        mask = data['mask']
        token_type_ids = data['token_type_ids']
        targets = data['targets']

        if device:
            ids = ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)

        optimizer.zero_grad()
        outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        batch_loss = loss.item()
        losses.append(batch_loss)
        if scheduler:
            scheduler.step()
    return np.mean(losses)

def eval_loop_fn(data_loader, model):
    model.eval()
    fin_targets = []
    fin_outputs = []
    losses = []
    device = config.DEVICE
    for batch_idx, data in enumerate(data_loader):
        ids = data['ids']
        mask = data['mask']
        token_type_ids = data['token_type_ids']
        targets = data['targets']

        if device:
            ids = ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)


        outputs =  model(ids=ids, mask=mask, token_type_ids=token_type_ids)
        loss = loss_fn(outputs, targets)
        losses.append(loss.item())
        fin_targets.append(targets.cpu().detach().numpy())
        fin_outputs.append(outputs.cpu().detach().numpy())
    return np.vstack(fin_outputs), np.vstack(fin_targets), np.mean(losses)


def run():
    data_df = pd.read_csv('../input/train.csv')
    train_df, valid_df = train_test_split(data_df, random_state=42, test_size=0.1)
    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)

    sample_sub_df = pd.read_csv("../input/sample_submission.csv")
    target_cols = list(sample_sub_df.drop("qa_id", axis=1).columns)
    train_y = train_df[target_cols].values
    valid_y = valid_df[target_cols].values

    train_dataset = BertDataset(
        qtitle=train_df.question_title.values,
        qbody=train_df.question_body.values,
        answer=train_df.answer.values,
        targets=train_y
    )

    valid_dataset = BertDataset(
        qtitle=valid_df.question_title.values,
        qbody=valid_df.question_body.values,
        answer=valid_df.answer.values,
        targets=valid_y
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size= config.TRAIN_BATCH_SIZE,
        shuffle=True
    )


    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size= config.VALID_BATCH_SIZE,
        shuffle=False
    )

    num_train_steps = int(len(train_dataset)/ config.TRAIN_BATCH_SIZE * config.EPOCHS)
    device= config.DEVICE
    model = BERTBaseUncased().to(device)
    optimizer = config.OPTIMIZER(model.parameters(), lr=config.LR)
    scheduler = config.SCHEDULER(optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)

    for epoch in range(config.EPOCHS):

        epoch_start = time.time()

        epoch_train_loss = train_loop_fn(train_dataloader, model, optimizer, scheduler)
        outputs, targets, epoch_valid_loss = eval_loop_fn(valid_dataloader, model)

        epoch_end = time.time()
        epoch_time_elapsed =  (epoch_end - epoch_start)/60.0
        print(f'time take to run a epoch - {epoch_time_elapsed}')
        print(f'Epoch - Training loss - {epoch_train_loss} Valid loss - {epoch_valid_loss}')

        spear =[]
        for jj in range(targets.shape[1]):
            p1 = list(targets[:, jj])
            p2 = list(outputs[:, jj])
            coef, _ = np.nan_to_num(stats.spearmanr(p1, p2))
            spear.append(coef)
        spear = np.mean(spear)
        print(f"Spearman coeff : {spear}")

if __name__ == '__main__':
    run()

