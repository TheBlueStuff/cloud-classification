import datetime
from tqdm import tqdm
import torch
import torch.nn.functional as F

import config


def train_fn(model, data_loader, cross_entropy_loss_1, cross_entropy_loss_2, optimizer):
    model.train()
    fin_loss = 0
    fin_preds = []
    fin_targs = []

    tk = tqdm(data_loader, total=len(data_loader))

    for data in tk:
        for k, v in data.items():
            data[k] = v.to(config.DEVICE)

        optimizer.zero_grad()
        logits_1, logits_2 = model(data["images"])
        loss_1 = cross_entropy_loss_1(logits_1, data["targets"])
        loss_2 = cross_entropy_loss_2(logits_2, data["targets"])
        loss = loss_1 + loss_2  # Can be multiplied by lambda factor
        loss.backward()
        optimizer.step()
        fin_loss += loss.item()

        batch_preds = F.softmax(logits_1, dim=-1)
        batch_preds = torch.argmax(batch_preds, dim=-1)

        fin_preds.append(batch_preds.cpu().numpy())
        fin_targs.append(data["targets"].cpu().numpy())

    return fin_preds, fin_targs, fin_loss / len(data_loader)


def eval_fn(model, data_loader, cross_entropy_loss_1, cross_entropy_loss_2):
    model.eval()
    fin_loss = 0
    fin_preds = []

    tk = tqdm(data_loader, total=len(data_loader))

    with torch.no_grad():
        for data in tk:
            for k, v in data.items():
                data[k] = v.to(config.DEVICE)

            logits_1, logits_2 = model(data["images"])
            loss_1 = cross_entropy_loss_1(logits_1, data["targets"])
            loss_2 = cross_entropy_loss_2(logits_2, data["targets"])
            loss = loss_1 + loss_2  # Can be multiplied by lambda factor
            fin_loss += loss.item()

            batch_preds = F.softmax(logits_1, dim=-1)
            batch_preds = torch.argmax(batch_preds, dim=-1)

            fin_preds.append(batch_preds.cpu().numpy())

    return (
        fin_preds,
        fin_loss / len(data_loader),
    )


def emb_fn(model, data_loader):
    model.eval()
    embeddings_list = []

    tk = tqdm(data_loader, total=len(data_loader))

    with torch.no_grad():
        for data in tk:
            for k, v in data.items():
                data[k] = v.to(config.DEVICE)

            embeds = model(
                data["images"],
                get_embeddings=True,
            )

            embeddings_list.append(embeds.cpu().numpy())

    return embeddings_list
