##############################################################################
# Name: download_sound_and_img_pairs.py
#
# - Downloads sound and image pairs from huggingface
# - https://huggingface.co/datasets/MVRL/iSoundNat
###############################################################################

import os
import itertools
import requests
import time
import threading
import ffmpeg
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # progress bar
from PIL import Image

##############################
# SETUP: DIRECTORIES & DATASET
##############################

mode = "train"  # or "validation" or "test"

# Define which split to use and CSV paths.
splits = {
    'train': 'train_df.csv',
    'validation': 'val_df.csv',
    'test': 'test_df.csv'
}
# Here we load the training CSV; adjust as needed.
df = pd.read_csv("hf://datasets/MVRL/iSoundNat/" + splits[mode])

# If you want to skip to a specific row (for example, row index 19000),
# then slice the DataFrame accordingly.
start_index = 0  
if start_index > 0:
    df = df.iloc[start_index:].reset_index(drop=True)

# Directories for saving images and audio files
image_save_dir = f"/mnt/hdd/inat2021_ds/sound_{mode}/images"
audio_save_dir = f"/mnt/hdd/inat2021_ds/sound_{mode}/sounds_mp3"
os.makedirs(image_save_dir, exist_ok=True)
os.makedirs(audio_save_dir, exist_ok=True)

# Convert dataframe rows to a list of dictionaries for iteration.
rows = df.to_dict("records")
num_rows = len(rows)

# Dictionaries to record failures for pairs (keyed by id)
image_failures = {}  
audio_failures = {}  

# Global progress bar and lock (one update per pair processed)
pbar = tqdm(total=num_rows, desc="Pairs Processed")
progress_lock = threading.Lock()

##############################
# HELPER FUNCTIONS
##############################

def convert_image_to_jpeg(temp_path, final_path):
    """
    Opens the image at temp_path using Pillow.
    If its format is not JPEG, converts it.
    Saves the image as JPEG to final_path.
    """
    try:
        with Image.open(temp_path) as im:
            if im.format != "JPEG":
                rgb_im = im.convert("RGB")
                rgb_im.save(final_path, "JPEG")
            else:
                # If already JPEG, simply rename the file.
                os.rename(temp_path, final_path)
    except Exception as e:
        print(f"Error converting image {temp_path}: {e}")
        raise e

def is_audio_corrupted(file_path):
    """
    Uses ffmpeg.probe() to check if an audio file is readable.
    Returns True if the file is corrupted or unreadable.
    """
    try:
        ffmpeg.probe(file_path)
        return False
    except ffmpeg.Error as e:
        print(f"Error probing audio '{file_path}': {e}")
        return True

def is_mp3_format(file_path):
    """
    Probes the file and checks whether 'mp3' is part of the format name.
    """
    try:
        info = ffmpeg.probe(file_path)
        format_name = info.get("format", {}).get("format_name", "")
        return "mp3" in format_name
    except Exception as e:
        print(f"Error checking mp3 format for '{file_path}': {e}")
        return False

def convert_to_mp3(input_file, output_file):
    """
    Converts the input audio file to MP3 using the libmp3lame codec.
    """
    try:
        stream = ffmpeg.input(input_file)
        stream = ffmpeg.output(stream, output_file, acodec="libmp3lame")
        ffmpeg.run(stream, quiet=True)
    except ffmpeg.Error as e:
        print(f"Error converting audio '{input_file}' to MP3: {e}")
        raise e

##############################
# DOWNLOAD FUNCTIONS WITH RETRIES
##############################

def download_image(row, image_url, image_id, image_save_path):
    """
    Downloads the image from row["image_url"].
    Saves a temporary file then converts (if needed) to JPEG as {id}.
    """

    temp_path = image_save_path + ".temp"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/115.0.0.0 Safari/537.36"
        )
    }
    max_retries = 3
    success = False
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(image_url, headers=headers, timeout=10)
            response.raise_for_status()
            with open(temp_path, "wb") as f:
                f.write(response.content)
            success = True
            break  # Exit loop on success.
        except Exception as e:
            if attempt < max_retries:
                time.sleep(2)
            else:
                print(f"FAILURE: Could not download image {image_id} from {image_url} after {max_retries} attempts")
                success = False
    if not success:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        return False

    try:
        convert_image_to_jpeg(temp_path, image_save_path)
    except Exception as e:
        print(f"Error processing image {image_id}: {e}")
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        if os.path.exists(image_save_path):
            try:
                os.remove(image_save_path)
            except Exception:
                pass
        return False
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

    return os.path.exists(image_save_path)

def download_audio(row, audio_url, audio_id, audio_save_path):
    """
    Downloads the audio file from row["sound_url"].
    Saves it to a temporary file, checks for corruption, and if needed converts it to MP3
    as {id}.mp3 using ffmpeg-python.
    """

    # temp_path = os.path.join(audio_save_dir, f"{audio_id}_temp")
    temp_path = audio_save_path + ".temp"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/115.0.0.0 Safari/537.36"
        )
    }
    max_retries = 5
    success = False
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(audio_url, headers=headers, timeout=10)
            response.raise_for_status()
            with open(temp_path, "wb") as f:
                f.write(response.content)
            success = True
            break
        except Exception as e:
            if attempt < max_retries:
                time.sleep(2)
            else:
                print(f"FAILURE: Could not download audio {audio_id} from {audio_url} after {max_retries} attempts")
                success = False
    if not success:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        return False

    # Check if the downloaded audio is corrupted.
    if is_audio_corrupted(temp_path):
        print(f"Audio file {audio_id} is corrupted.")
        try:
            os.remove(temp_path)
        except Exception:
            pass
        return False

    # Check if the audio is already in MP3 format.
    if is_mp3_format(temp_path):
        try:
            os.rename(temp_path, audio_save_path)
        except Exception as e:
            print(f"Error renaming audio {audio_id}: {e}")
            try:
                os.remove(temp_path)
            except Exception:
                pass
            return False
        return True
    else:
        try:
            convert_to_mp3(temp_path, audio_save_path)
        except Exception as e:
            print(f"Error converting audio {audio_id}: {e}")
            try:
                os.remove(temp_path)
            except Exception:
                pass
            return False
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

    return os.path.exists(audio_save_path)

def download_pair(row):
    """
    Downloads both the image and audio for a given row.
    If either download/conversion fails, deletes any successfully downloaded file
    and marks the pair as a failure.
    """

    # If the final image already exists, assume it's already downloaded.
    image_url = row["image_url"]
    image_id = row["id"]
    img_save_path = os.path.join(image_save_dir, f"{image_id}.jpg")
    
    img_exists = False
    if os.path.exists(img_save_path):
        img_exists = True
    

    # If the final audio already exists, assume it's already downloaded.
    audio_url = row["sound_url"]
    audio_id = row["id"]
    audio_save_path = os.path.join(audio_save_dir, f"{audio_id}.mp3")
    
    audio_exists = False
    if os.path.exists(audio_save_path):
        audio_exists = True

    # Skip the download if both files already exist.
    if not (img_exists and audio_exists):

        image_success = download_image(row, image_url, image_id, img_save_path)
        audio_success = download_audio(row, audio_url, audio_id, audio_save_path)

        # If either download failed, delete any successfully downloaded file.
        if not (image_success and audio_success):
            image_failures[row["id"]] = row["image_url"]
            audio_failures[row["id"]] = row["sound_url"]
            success = False
        else:
            success = True
    else:
        success = True
        print(f"SKIPPED: Image {image_id} and Audio {audio_id} already exists.")

    with progress_lock:
        pbar.update(1)
    return success

def chunked_iterator(iterable, chunk_size):
    """
    Yields successive chunks of size chunk_size from the iterable.
    """
    iterator = iter(iterable)
    while True:
        chunk = list(itertools.islice(iterator, chunk_size))
        if not chunk:
            break
        yield chunk

##############################
# PROCESS THE DATASET IN CHUNKS
##############################

chunk_size = 999999   # Adjust based on memory and dataset size.
max_workers = 8       # Number of threads for parallel downloads.

# Process rows in chunks using multi-threading.
try:
    for chunk in chunked_iterator(rows, chunk_size):
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(download_pair, row): row for row in chunk}
            for future in as_completed(futures):
                try:
                    future.result()  # True if both downloads succeeded.
                except Exception as e:
                    row = futures[future]
                    print(f"Error processing row {row['id']}: {e}")
except Exception as e:
    print(f"An error occurred during processing: {e}")

pbar.close()

print("Image download failures:", image_failures)
print("Audio download failures:", audio_failures)
print("len(image_failures):", len(image_failures))
print("len(audio_failures):", len(audio_failures))

##############################
# REMOVE FAILURE ROWS FROM ORIGINAL DATAFRAME AND EXPORT
##############################

# Combine IDs from both failure dictionaries.
failure_ids = set(image_failures.keys()).union(set(audio_failures.keys()))
print(f"Total failed pairs: {len(failure_ids)}")

# Remove failed rows from the original dataframe (preserving original order).
successful_df = df[~df["id"].isin(failure_ids)]

output_csv = f"/mnt/hdd/inat2021_ds/sound_{mode}/sound_image_pairs_filtered.csv"
successful_df.to_csv(output_csv, index=False)
print(f"Exported {len(successful_df)} successful rows to {output_csv}")
