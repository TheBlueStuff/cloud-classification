from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np

from . import config


def train_fn(model, data_loader, criterion, optimizer):
    model.train()
    fin_loss = 0
    fin_preds = []
    fin_targs = []

    for data in tqdm(data_loader, total=len(data_loader)):
            
        if 'augmented' in data.keys():
            images = torch.cat([data["images"], data['augmented']],dim=0).to(config.DEVICE)
            targets = data["targets"].repeat(2).to(config.DEVICE)
            
        else:
            images = data["targets"].to(config.DEVICE)
            targets = data["targets"].to(config.DEVICE)

        optimizer.zero_grad()
        
        logits = model(images)
        
        _loss = criterion(logits, targets)
        _loss.backward()
        
        optimizer.step()
        
        fin_loss += _loss.item()

        batch_preds = F.softmax(logits, dim=1)
        batch_preds = torch.argmax(batch_preds, dim=1)

        fin_preds.append(batch_preds.cpu().numpy())
        fin_targs.append(targets.cpu().numpy())

    return (np.concatenate(fin_preds,axis=0), 
            np.concatenate(fin_targs,axis=0), 
            fin_loss / len(data_loader))


def eval_fn(model, data_loader, criterion):
    model.eval()
    fin_loss = 0
    fin_preds = []
    fin_targs = []

    with torch.no_grad():
        for data in tqdm(data_loader, total=len(data_loader)):
            for k, v in data.items():
                data[k] = v.to(config.DEVICE)

            logits = model(data["images"])
            _loss = criterion(logits, data["targets"])
            fin_loss += _loss.item()

            batch_preds = F.softmax(logits, dim=1)
            batch_preds = torch.argmax(batch_preds, dim=1)

            fin_preds.append(batch_preds.cpu().numpy())
            fin_targs.append(data["targets"].cpu().numpy())

    return (
        np.concatenate(fin_preds,axis=0),
        np.concatenate(fin_targs,axis=0),
        fin_loss / len(data_loader),
    )