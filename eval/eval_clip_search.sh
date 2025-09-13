#!/bin/bash

# For running CLIP AVS with and without TTA (Image / Text) + RL Planner

cd ../
export POLICY="RL"
export NUM_EPS_STEPS=256
export FIX_START_POSITION=False
export USE_CLIP_PREDS=True
export QUERY_MODALITY="image"   # "image", "text"
export NUM_GPU=1
export NUM_META_AGENT=2
export SAVE_GIFS=False
export AVS_LOAD_PRETRAINED_HF_CHECKPOINT=True
export AVS_SAT_CHECKPOINT_PATH=""      # If loading locally

############################################

# Expt 1: ALL Val IN (w/ TTA)
export EXECUTE_TTA=True
export OVERRIDE_MASK_DIR=""
export QUERY_TAX=""
export AVS_SAT_TO_IMG_IDS_PATH="search_eval_trimodal|val_in_domain"

export CSV_EXPT_NAME="${POLICY}_ALL_val_in_TTA_${EXECUTE_TTA}_modality_${QUERY_MODALITY}_steps_${NUM_EPS_STEPS}"
python -m planner.test_driver

############################################

# Expt 2: ALL Val IN (w/o TTA)
export EXECUTE_TTA=False
export OVERRIDE_MASK_DIR=""
export QUERY_TAX=""
export AVS_SAT_TO_IMG_IDS_PATH="search_eval_trimodal|val_in_domain"

export CSV_EXPT_NAME="${POLICY}_ALL_val_in_TTA_${EXECUTE_TTA}_modality_${QUERY_MODALITY}_steps_${NUM_EPS_STEPS}"
python -m planner.test_driver

############################################

# Expt 3: ALL Val OUT (w/ TTA)
export EXECUTE_TTA=True
export OVERRIDE_MASK_DIR=""
export QUERY_TAX=""
export AVS_SAT_TO_IMG_IDS_PATH="search_eval_trimodal|val_out_domain"

export CSV_EXPT_NAME="${POLICY}_ALL_val_out_TTA_${EXECUTE_TTA}_modality_${QUERY_MODALITY}_steps_${NUM_EPS_STEPS}"
python -m planner.test_driver

############################################

# Expt 4: ALL Val OUT (w/o TTA)
export EXECUTE_TTA=False
export OVERRIDE_MASK_DIR=""
export QUERY_TAX=""
export AVS_SAT_TO_IMG_IDS_PATH="search_eval_trimodal|val_out_domain"

export CSV_EXPT_NAME="${POLICY}_ALL_val_out_TTA_${EXECUTE_TTA}_modality_${QUERY_MODALITY}_steps_${NUM_EPS_STEPS}"
python -m planner.test_driver
