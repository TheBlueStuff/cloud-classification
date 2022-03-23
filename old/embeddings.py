import random
import os
import glob
import pandas as pd
import numpy as np
import torch

import config
import dataset
import engine

from model import GATClassifier


def get_embeddings():

    print("Preparando data...")

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

    # print(len(train_img_names), len(test_img_names))

    test_dataset = dataset.ImageClassificationDataset(
        image_paths=train_image_files + test_image_files,
        targets=train_targets + test_targets,
        normalization=(config.IMAGE_MEAN, config.IMAGE_STD),
        resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        shuffle=False,
    )

    # model config
    print("Configurando Modelo..")

    model = GATClassifier(
        in_channels=3,
        emb_dims=2048,  # Deep features
        in_dims=256,
        out_dims=512,  # Graph features
        num_classes=7,
    )

    # Load params
    model.load_model("./model params/model_params_1")

    model.to(config.DEVICE)

    print("Empezando Procesamiento..")
    best_acc = 0

    batch_embeddings = engine.emb_fn(model, test_loader)

    embeddings = []

    # Test preds
    for i, vp in enumerate(batch_embeddings):
        embeddings.extend(vp)

    df = pd.DataFrame(
        {
            "filename": train_img_names + test_img_names,
            "targets": train_targets + test_targets,
        }
    )

    df.to_csv("embeddings.csv", encoding="utf-8", index=False)
    np.savez_compressed("embeddings.npz", embeddings)


if __name__ == "__main__":
    get_embeddings()
