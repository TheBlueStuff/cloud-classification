import os
import glob
import torch
import torch.nn as nn
import numpy as np
import random

from sklearn import metrics

import config
import dataset
import engine
from torch.utils.tensorboard import SummaryWriter

from model import GATClassifier


def run_training():
    train_image_files = glob.glob(
        os.path.join(config.DATA_DIR, "GCD/train/**/*.jpg"), recursive=True
    )

    train_targets = [x.split("\\")[-1][0] for x in train_image_files]
    train_targets = list(map(int, train_targets))  # parse to int
    train_targets = [i - 1 for i in train_targets]

    train_img_names = [
        x.split("\\")[-1] for x in train_image_files
    ]  # Classes go from 1 to 7, shift 0-6

    test_image_files = glob.glob(
        os.path.join(config.DATA_DIR, "GCD/test/**/*.jpg"), recursive=True
    )

    test_targets = [x.split("\\")[-1][0] for x in test_image_files]
    test_targets = list(map(int, test_targets))  # parse to int
    test_targets = [i - 1 for i in test_targets]

    test_img_names = [x.split("\\")[-1] for x in test_image_files]

    train_dataset = dataset.ImageClassificationDataset(
        image_paths=train_image_files,
        targets=train_targets,
        normalization=(config.IMAGE_MEAN, config.IMAGE_STD),
        resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
    )

    test_dataset = dataset.ImageClassificationDataset(
        image_paths=test_image_files,
        targets=test_targets,
        normalization=(config.IMAGE_MEAN, config.IMAGE_STD),
        resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=False,
    )

    # Model setup and optim
    model = GATClassifier(
        in_channels=3,
        emb_dims=2048,  # Deep features
        in_dims=256,
        out_dims=512,  # Graph features
        num_classes=7,
    )

    model.to(config.DEVICE)

    cross_entropy_loss_1 = nn.CrossEntropyLoss()
    cross_entropy_loss_2 = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    # TensorBoard
    # writer = SummaryWriter()
    print("Empezando Entrenamiento..")
    best_acc = 0

    for epoch in range(config.EPOCHS):
        batch_train_preds, batch_train_targs, train_loss = engine.train_fn(
            model, train_loader, cross_entropy_loss_1, cross_entropy_loss_2, optimizer
        )

        (batch_preds, test_loss,) = engine.eval_fn(
            model, test_loader, cross_entropy_loss_1, cross_entropy_loss_2
        )

        train_preds = []
        train_targs = []
        epoch_preds = []

        # train
        for i, vp in enumerate(batch_train_preds):
            train_preds.extend(vp)
            train_targs.extend(batch_train_targs[i])

        # test
        for i, vp in enumerate(batch_preds):
            epoch_preds.extend(vp)

        train_accuracy = metrics.accuracy_score(train_targs, train_preds)
        test_accuracy = metrics.accuracy_score(test_targets, epoch_preds)

        print(
            f"Epoch={epoch+1}, Train Loss={train_loss:.4f}, Train Accuracy={train_accuracy:.4f}, Test Loss={test_loss:.4f} Test Accuracy={test_accuracy:.4f}"
        )

        # print("Writing to TensorBoard..")
        # writer.add_scalar("Loss/train", train_loss, epoch)
        # writer.add_scalar("Accuracy/train", train_accuracy, epoch)
        # writer.add_scalar("Loss/test", test_loss, epoch)
        # writer.add_scalar("Accuracy/test", test_accuracy, epoch)

        # Save model
        if test_accuracy > best_acc:
            model.save_model(f"model params/model_params_5")
            best_acc = test_accuracy


if __name__ == "__main__":
    run_training()
