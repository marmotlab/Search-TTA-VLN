#!/bin/bash

# For running pre-generated masks for AVS + RL Planner

cd ../
export POLICY="RL"
export NUM_EPS_STEPS=256
export FIX_START_POSITION=False
export USE_CLIP_PREDS=False
export EXECUTE_TTA=False
export NUM_GPU=1
export NUM_META_AGENT=8
export SAVE_GIFS=False

#======= LLMSeg =======#

# Expt 1: ALL (Val_in)
export OVERRIDE_MASK_DIR="maps/lisa/masks_val_in"
export QUERY_TAX=""
export AVS_SAT_TO_IMG_IDS_PATH="search_eval_trimodal|val_in_domain"

export CSV_EXPT_NAME="LISA_${POLICY}_all_val_in_steps_${NUM_EPS_STEPS}"
python -m planner.test_driver

############################################

# Expt 2: ALL (Val_out)
export OVERRIDE_MASK_DIR="maps/lisa/masks_val_out"
export QUERY_TAX=""
export AVS_SAT_TO_IMG_IDS_PATH="search_eval_trimodal|val_out_domain"

export CSV_EXPT_NAME="LISA_${POLICY}_all_val_out_TTA_steps_${NUM_EPS_STEPS}"
python -m planner.test_driver

############################################