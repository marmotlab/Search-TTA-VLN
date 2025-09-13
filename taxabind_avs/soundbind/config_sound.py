##############################################################################
# Name: config_sound.py
#
# - Parameters for train/eval sound encoder
###############################################################################

from easydict import EasyDict as edict

config = edict()
config.train_df = 'clip_train_quadrimodal|train'
config.val_df = 'clip_train_quadrimodal|val'
config.data_path_train = '/mnt/hdd/avs_bench_ds/sound_mp3/train'
config.data_path_val = '/mnt/hdd/avs_bench_ds/sound_mp3/test'

config.batch_size = 256
config.lr = 1e-4
config.accumulate_grad_batches = 8
config.max_epochs = 20
config.num_workers = 16
config.devices = 2
config.val_check_interval = 0.5
config.sound_encoder = 'laion/clap-htsat-fused'
config.avs_dataset = 'derektan95/avs-bench'
config.save_dir = 'checkpoints'
config.filename = 'soundbind-{epoch:02d}-{val_loss:.2f}'
config.locked_tuning = True

# huggingface finetuned
config.image_encoder_finetuned = 'hf-hub:imageomics/bioclip'    

print("config: \n", config)