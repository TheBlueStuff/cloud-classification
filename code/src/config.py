
DATA_DIR = '/data/mandonaire'
DEVICE = 'cuda:2'

EPOCHS = 25
BATCH_SIZE = 32


SAVE_PATH = '/data/ltorres/model_parameters'

GNN_LOSS_LAMBDA = 1 # <1


########### STRUCTURES ##########

AUGMENTATION_TYPES = [
    'hflip',
    'v_flip',
    'rot',
    'r_crop',
]
