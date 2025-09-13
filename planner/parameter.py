#######################################################################
# Name: parameter.py
#
# NOTE: Change all your hyper-params here for training RL policy.
#######################################################################

import sys
sys.modules['TRAINING'] = True          


# --- GENERAL --- #
USE_GPU = False             # Collect training data using GPUs
USE_GPU_GLOBAL = True       # Train the network using GPUs
NUM_GPU = 1
NUM_META_AGENT = 8          # Number of concurrent episodes (i.e. workers)
NUM_EPS_STEPS=384           # 256, 384
NUM_ROBOTS_MIN=1            # Currently only designed for 1 robot
NUM_ROBOTS_MAX=1
NUM_COORDS_WIDTH=24         # How many node coords across width?
NUM_COORDS_HEIGHT=24        # How many node coords across height?
SENSOR_RANGE=80             # Only applicable to 'circle' sensor model
SENSOR_MODEL="rectangular"  # "rectangular", "circle" (NOTE: no colllision check for rectangular)
TERMINATE_ON_TGTS_FOUND = False        # Whether to terminate episode when all targets found
FORCE_LOGGING_DONE_TGTS_FOUND = False  # Whether to force csv logging when all targets found
FIX_START_POSITION = False             # Whether to fix the starting position of the robots (bottom left corner)


# --- RL Training Params --- # 
LOAD_MODEL = False
RESET_LOG_ALPHA = True
LOG_ALPHA_START = -2.0
MODEL_NAME = "checkpoint.pth"
REPLAY_SIZE = 10000
MINIMUM_BUFFER_SIZE = 2000
BATCH_SIZE = 128 
INPUT_DIM = 4
EMBEDDING_DIM = 128
NODE_PADDING_SIZE = 580     # More than NUM_COORDS_WIDTH*NUM_COORDS_HEIGHT
K_SIZE = 8                  # Nearest neighbour edges
LR = 1e-5
GAMMA = 0.995
DECAY_STEP = 256
SUMMARY_WINDOW = 32         # Logs to tensorboard
LOAD_AVS_BENCH = False      # Not used during training
GPU_RESOURCE_REQUEST = None # Not manually specifying Ray GPU allocation
GPU_RAY_MAPPING = None      # Not manually specifying Ray GPU allocation


# --- Folders & Visualizations --- #
GRIDMAP_SET_DIR = "maps/gpt4o/envs_train"
MASK_SET_DIR = "maps/gpt4o/masks_train"
TARGETS_SET_DIR = ""        # Whether tgts are provided by external files 
MASKS_RAND_INDICES_PATH = "maps/gpt4o/masks_train_indices.json"
FOLDER_NAME = 'avs_search' 
MODEL_PATH = f'train/model/{FOLDER_NAME}'
TRAIN_PATH = f'train/logs/{FOLDER_NAME}'
GIFS_PATH = f'train/gifs/{FOLDER_NAME}'
SAVE_IMG_GAP = 101
VIZ_GRAPH_EDGES = True


# COLORS (for printing)
RED='\033[1;31m'          
GREEN='\033[1;32m'
YELLOW='\033[1;93m'       
NC_BOLD='\033[1m' # Bold, No Color 
NC='\033[0m' # No Color 
