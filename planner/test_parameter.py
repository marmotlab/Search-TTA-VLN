############################################################################################
# Name: test_parameter.py
#
# NOTE: Change all your hyper-params here for eval
# Simple How-To Guide: 
# 1. CLIP TTA: USE_CLIP_PREDS=True, EXECUTE_TTA=True
# 2. CLIP (No TTA): USE_CLIP_PREDS=True, EXECUTE_TTA=False 
# 3. Custom masks (e.g. LISA): USE_CLIP_PREDS=False, EXECUTE_TTA=False 
############################################################################################

import os
import sys
sys.modules['TRAINING'] = False           # False = Inference Testing    

###############################################################
# Overload Params
###############################################################

OPT_VARS = {}
def getenv(var_name, default=None, cast_type=str):
    try:
        value = os.environ.get(var_name, None)
        if value is None:
            result = default
        elif cast_type == bool:
            result = value.lower() in ("true", "1", "yes")
        else:
            result = cast_type(value)
    except (ValueError, TypeError):
        result = default

    OPT_VARS[var_name] = result  # Log the result
    return result

###############################################################
# General
###############################################################

# --- GENERAL --- #
USE_GPU = True  
NUM_GPU = getenv("NUM_GPU", default=1, cast_type=int) # the number of GPUs
NUM_META_AGENT = getenv("NUM_META_AGENT", default=2, cast_type=int)  # the number of concurrent processes
NUM_EPS_STEPS = getenv("NUM_EPS_STEPS", default=384, cast_type=int)
FIX_START_POSITION = getenv("FIX_START_POSITION", default=True, cast_type=bool)  # Whether to fix the starting position of the robots (middle index)
NUM_ROBOTS = 1                        # Only allow for 1 robot
NUM_COORDS_WIDTH=24                   # How many node coords across width?
NUM_COORDS_HEIGHT=24                  # How many node coords across height?
CLIP_GRIDS_DIMS=[24,24]               # [16,16] if 'openai/clip-vit-large-patch14-336' 
SENSOR_RANGE=80                       # Only applicable to 'circle' sensor model
SENSOR_MODEL="rectangular"            # "rectangular", "circle" (NOTE: no colllision check for rectangular)
TERMINATE_ON_TGTS_FOUND = False       # Whether to terminate episode when all targets found
FORCE_LOGGING_DONE_TGTS_FOUND = True  # Whether to force csv logging when all targets found


# --- Planner Params --- # 
POLICY = getenv("POLICY", default="RL", cast_type=str)
NUM_TEST = 800  # Overriden if LOAD_AVS_BENCH 
NUM_RUN = 1
MODEL_NAME = "avs_rl_policy.pth" 
INPUT_DIM = 4
EMBEDDING_DIM = 128
K_SIZE = 8   


# --- Folders & Visualizations --- #
GRIDMAP_SET_DIR = "maps/gpt4o/envs_val"
MASK_SET_DIR = "maps/example/masks_val"      # Overriden if LOAD_AVS_BENCH
TARGETS_SET_DIR = ""
# TARGETS_SET_DIR = "maps/example/gt_masks_val_with_tgts"   # Overriden if LOAD_AVS_BENCH
OVERRIDE_MASK_DIR = getenv("OVERRIDE_MASK_DIR", default="", cast_type=str)  # Override initial score mask from CLIP 
SAVE_GIFS = getenv("SAVE_GIFS", default=False, cast_type=bool)              # do you want to save GIFs
FOLDER_NAME = 'avs_search' 
MODEL_PATH = f'inference/model'
GIFS_PATH = f'inference/test_results/gifs/{FOLDER_NAME}'
LOG_PATH = f'inference/test_results/log/{FOLDER_NAME}'
LOG_TEMPLATE_XLSX = f'inference/template.xlsx'
CSV_EXPT_NAME = getenv("CSV_EXPT_NAME", default="data", cast_type=str)
VIZ_GRAPH_EDGES = False  # do you want to visualize the graph edges


#######################################################################
# AVS Params
#######################################################################

# General PARAMS
USE_CLIP_PREDS = getenv("USE_CLIP_PREDS", default=True, cast_type=bool)   # If false, use custom masks from OVERRIDE_MASK_DIR
QUERY_TAX = getenv("QUERY_TAX", default="", cast_type=str)                # "" = Test all tax (can accept taxonomy substrings)
EXECUTE_TTA = getenv("EXECUTE_TTA", default=True, cast_type=bool)         # Whether to execute TTA mask updates
QUERY_MODALITY = getenv("QUERY_MODALITY", default="image", cast_type=str) # "image", "text", "sound"
STEPS_PER_TTA = 20  # no. steps before each TTA series
NUM_TTA_STEPS = 1   # no. of TTA steps during each series
RESET_WEIGHTS = True
MIN_LR = 1e-6
MAX_LR = 1e-5   
GAMMA_EXPONENT = 2  

# Paths related to AVS (TRAIN w/ TARGETS)
LOAD_AVS_BENCH = True     # Whether to init AVS datasets 
AVS_IMG_DIR = '/mnt/hdd/avs_bench_ds/inat21' 
AVS_IMO_DIR = '/mnt/hdd/avs_bench_ds/sat_jpg/train_512px' 
AVS_INAT_JSON_PATH = '/mnt/hdd/avs_bench_ds/inat21/train.json' 
AVS_SOUND_DIR = '/mnt/hdd/avs_bench_ds/sound_mp3/test'
AVS_GAUSSIAN_BLUR_KERNEL = (5,5)
AVS_SAT_TO_IMG_IDS_PATH = getenv("AVS_SAT_TO_IMG_IDS_PATH", default="search_eval_trimodal|val_in_domain", cast_type=str)
AVS_LOAD_PRETRAINED_HF_CHECKPOINT = getenv("AVS_LOAD_PRETRAINED_HF_CHECKPOINT", default=True, cast_type=bool)  # If false, load locally using CHECKPOINT_PATHs
AVS_SAT_CHECKPOINT_PATH = getenv("AVS_SAT_CHECKPOINT_PATH", default="", cast_type=str)                  
AVS_SOUND_CHECKPOINT_PATH = getenv("AVS_SOUND_CHECKPOINT_PATH", default="", cast_type=str)   

#######################################################################
# UTILS
#######################################################################

# COLORS (for printing)
RED='\033[1;31m'          
GREEN='\033[1;32m'
YELLOW='\033[1;93m'       
NC_BOLD='\033[1m' # Bold, No Color 
NC='\033[0m'      # No Color 
