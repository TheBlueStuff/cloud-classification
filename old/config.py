import torch

DATA_DIR = "../../dataset/"
BATCH_SIZE = 32
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256

# Calculated values for train images
IMAGE_MEAN = 155.5673
IMAGE_STD = 70.5983

NUM_WORKERS = 4
EPOCHS = 20
NUM_EMBEDDINGS = 1000
THRESHOLD = 0.75
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
