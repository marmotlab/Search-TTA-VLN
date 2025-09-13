##############################################################################
# Name: config_sat.py
#
# - Parameters for train/eval sat encoder
###############################################################################

from easydict import EasyDict as edict     

config = edict()

# Pixel level CLIP training
config.img_dir = '/mnt/hdd/avs_bench_ds/inat21' 
config.imo_dir = '/mnt/hdd/avs_bench_ds/sat_jpg/train_512px'   
config.imo_dir_val = '/mnt/hdd/avs_bench_ds/sat_jpg/test_512px'    
config.train_json_path = '/mnt/hdd/avs_bench_ds/inat21/train.json' 
config.val_json_path = '/mnt/hdd/avs_bench_ds/inat21/val.json' 
config.sat_to_img_ids_train_json_path = 'clip_train_trimodal|train'
config.sat_to_img_ids_val_json_path = 'clip_train_trimodal|val'

# batch_size * accumulate_grad_batches * devices = constant (i.e. 256 * 8 * 2 = 4096)
config.batch_size = 32 
config.lr = 1e-4    
config.accumulate_grad_batches = 64
config.max_epochs = 20
config.num_workers = 16
config.devices = 2 
config.val_check_interval = 0.5
config.sat_encoder = 'openai/clip-vit-large-patch14-336' 
config.avs_dataset = 'derektan95/avs-bench'
config.patch_size = 14

config.save_dir = 'checkpoints'
config.filename = 'satbind-{epoch:02d}-{val_loss:.2f}'

config.locked_tuning = True

config.resume_from_checkpoint = False
config.resume_checkpoint_name = 'satbind-resume'

# huggingface finetuned
config.image_encoder_finetuned = 'hf-hub:imageomics/bioclip'
config.sound_encoder_finetuned = 'derektan95/search-tta-sound'  # For eval only
config.sat_encoder_finetuned = 'derektan95/search-tta-sat'      # For eval only

print("config: \n", config)