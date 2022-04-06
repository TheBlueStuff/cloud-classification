
DATA_DIR = '/data/mandonaire'
DEVICE = 'cuda:2'

EPOCHS = 30
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

GCD_CLASSES = [
    '1_cumulus',
    '2_altocumulus',
    '3_cirrus',
    '4_clearsky',
    '5_stratocumulus',
    '6_cumulonimbus',
    '7_mixed'
]