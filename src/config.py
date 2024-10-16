import os, sys
from config import *
import argparse
from typing import List, Tuple, Optional

RANDOM_SEED = 10001
NUM_CPU = 32

MAX_NUM_FRAG = 1024
WANTED_RESOLUTION = 0  # 1024 512 256 128 64 32, if the resolution is 0, that means the original resolution will be saved


DIR_SAVE_FULL_HICMX_HUMAN = "dataset/full_hic_madeup"
DIR_SAVE_TRAIN_DATA_HUMAN ="dataset/train_data_madeup"

DIR_SAVE_FULL_HICMX_HUMAN_TEST = "dataset/full_hic_madeup_test"
DIR_SAVE_TRAIN_DATA_HUMAN_TEST ="dataset/train_data_madeup_test"



WORKSPACE_PATH = "/lustre/scratch125/ids/team117-assembly/hackathon"

# ===============================
# random seed
RANDOM_SEED = 10001

# ===============================
# training settings
PAD_TOKEN = 0
TRAIN_SET_RATIO = 0.8
N_EPOCHS = 1000
SHOW_PER_EPOCH = 10
N_CHANNEL = 4
START_TOKEN = 2 * MAX_NUM_FRAG + 1
NUM_TOKEN = 2 * MAX_NUM_FRAG + 2
N_HIDDEN = 64
BATCH_SIZE = 32

# ===============================
# network settings
D_MODEL = 512
NUM_CHANNELS = NUM_HEADS = 4  # setting for hic-attention transformer
NUM_LAYERS = 2
D_FF = 2048
MAX_SEQ_LENGTH = MAX_NUM_FRAG
DROPOUT = 0.1


# parse the input from Command-Line
parser = argparse.ArgumentParser(description="A script to specify the parameters for training Sorting NN.")
parser.add_argument('-n', '--num_epochs', type=int, help='Number of epochs to run on this training round')
parser.add_argument('-r', '--random_seed', type=int, help='Random seed for this runing')
parser.add_argument('-b', '--batch_size', type=int, help='Batch size for this runing')
parser.add_argument('-s', '--show_per_epoch', type=int, help='Show the training loss per epoch')
parser.add_argument('-m', '--madeup_data_include_flag', action='store_true', help='Whether to include the madeup data in the training')
parser.add_argument('-c', '--curated_data_flag', type=int, help='Whether to only use the madeup data in the training')
args, unknown = parser.parse_known_args()

print(f"Warning: Unknown args: " + ", ".join(unknown), file=sys.stderr) 

if args.num_epochs is not None:
    if args.num_epochs >=1:
        N_EPOCHS = args.num_epochs
        print(f"Update N_EPOCHS: {N_EPOCHS}")
    else:
        print(f"The num_epochs ({args.num_epochs}) is invalid. N_EPOCHS stays: {N_EPOCHS}")
if args.random_seed is not None:
    RANDOM_SEED = args.random_seed
    print(f"Update RANDOM_SEED: {RANDOM_SEED}")
if args.batch_size is not None:
    if args.batch_size >=1 and args.batch_size <= 2048:
        BATCH_SIZE = args.batch_size
        print(f"Update BATCH_SIZE: {BATCH_SIZE}")
    else:
        print(f"The batch_size ({args.batch_size}) is invalid. BATCH_SIZE stays: {BATCH_SIZE}")
if args.show_per_epoch is not None:
    if args.show_per_epoch >=1 and args.show_per_epoch <= N_EPOCHS:
        SHOW_PER_EPOCH = args.show_per_epoch
        print(f"Update SHOW_PER_EPOCH: {SHOW_PER_EPOCH}")
    else:
        print(f"The batch_size ({args.show_per_epoch}) is invalid. SHOW_PER_EPOCH stays: {SHOW_PER_EPOCH}")

if args.madeup_data_include_flag is not None:
    MADEUP_DATA_INCLUDE_FLAG = args.madeup_data_include_flag
    print(f"Update MADEUP_DATA_INCLUDE_FLAG: {MADEUP_DATA_INCLUDE_FLAG}")
if args.curated_data_flag is not None:
    if args.curated_data_flag==0:
        CURATED_DATA_FLAG = False
    print(f"Update CURATED_DATA_FLAG: {CURATED_DATA_FLAG}")


# ===============================
# model save path & predicted likelihood matrix save path
PATH_SAVE_CHECHPOINT_GRAGPH_LIKELIHOOD = os.path.join(WORKSPACE_PATH, "trained_model/likelihood_graph_madeup")
PATH_SAVE_PREDICTED_LIKELIHOOD_GRAPH = os.path.join(WORKSPACE_PATH,  "results/predicted_likelihood_graph_madeup")
PATH_SAVE_PREDICTED_LIKELIHOOD_GRAPH_TEST = os.path.join(WORKSPACE_PATH,  "results/predicted_likelihood_graph_madeup_test")


