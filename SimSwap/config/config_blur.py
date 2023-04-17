import os
DIR_HOME = os.path.expanduser("~")
PROJECT_DIR = os.path.join(DIR_HOME, 'Projects/ProFaceInv')


# Training parameters
recognizer = 'IResNet100'
obfuscator = 'blur_21_2_6'
dataset_dir = os.path.join(DIR_HOME, 'Datasets/CelebA/align_crop_224')
eval_dir = os.path.join(DIR_HOME, 'Datasets/LFW/LFW_112_test_pairs')
eval_pairs = os.path.join(DIR_HOME, 'Datasets/LFW/pairs.txt')
debug = False

# Path to saved checkpoints
MODEL_PATH = [
    # os.path.join(PROJECT_DIR, 'checkpoints/pixelate_4_10_IResNet100_ep3_iter16600.pth'),
    # os.path.join(PROJECT_DIR, 'checkpoints/pixelate_4_10_IResNet100_ep3_iter7000.pth'),
    # os.path.join(PROJECT_DIR, 'checkpoints/pixelate_4_10_IResNet100_ep2_iter9000.pth'),
    # os.path.join(PROJECT_DIR, 'checkpoints/pixelate_4_10_IResNet100_ep1_iter3000.pth')
]

# Super parameters
clamp = 2.0
channels_in = 3
# log10_lr = -3.5 #-4.5
# lr = 10 ** log10_lr
lr = 0.000125
epochs = 1000
weight_decay = 1e-5
init_scale = 0.01

lamda_reconstruction = 5
lamda_guide = 1
lamda_low_frequency = 1
device_ids = [0, 1]

# Train:
batch_size = 16
cropsize = 224
betas = (0.5, 0.999)
weight_step = 1000
gamma = 0.5

# Val:
cropsize_val = 1024
batchsize_val = 2
shuffle_val = False
val_freq = 50


# Dataset
TRAIN_PATH = '/home/jjp/Dataset/DIV2K/DIV2K_train_HR/'
VAL_PATH = '/home/jjp/Dataset/DIV2K/DIV2K_valid_HR/'
format_train = 'png'
format_val = 'png'

# Display and logging:
loss_display_cutoff = 2.0
loss_names = ['L', 'lr']
silent = False
live_visualization = False
progress_bar = False


# Saving checkpoints:

# MODEL_PATH = '/home/jjp/Hinet/model/'
checkpoint_on_error = True
SAVE_freq = 50

IMAGE_PATH = '/home/jjp/Hinet/image/'
IMAGE_PATH_cover = IMAGE_PATH + 'cover/'
IMAGE_PATH_secret = IMAGE_PATH + 'secret/'
IMAGE_PATH_steg = IMAGE_PATH + 'steg/'
IMAGE_PATH_secret_rev = IMAGE_PATH + 'secret-rev/'

# Load:
suffix = 'model.pt'
tain_next = False
trained_epoch = 0
