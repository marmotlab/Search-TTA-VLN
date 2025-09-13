##############################################################################
# Name: download_sat_imgs.py
#
# - Downloads satellite images and relevant data from huggingface
# - https://huggingface.co/datasets/MVRL/iSatNat
###############################################################################

import os
import itertools
import requests
import time
import threading
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  
from PIL import Image, UnidentifiedImageError
from io import BytesIO

# Load the dataset from Hugging Face
mode = "train"  # train or test
ds = load_dataset("MVRL/iSatNat", split=f"{mode}")
num_rows = len(ds)  
resize_size = 512  # Resize images to this size (i.e. same FOV - increase resolution)

# Directory where images will be saved
sat_save_dir = f"/mnt/hdd/inat2021_ds/sat_{mode}"
os.makedirs(sat_save_dir, exist_ok=True)

# Dictionary to store download failures: {key: sat_url}
download_failures = {}

# Create a global progress bar for images processed.
pbar = tqdm(total=num_rows, desc="Images Processed")    

# Create a lock for thread-safe updates to the progress bar.
progress_lock = threading.Lock()

def download_image(row):
    """
    Download the image from row['sat_url'] and save it as sat_save_dir/{row['key']}.jpeg.
    If the download fails, it will retry up to 3 times before recording a failure.
    Each attempt prints a success or retry message.
    The progress bar is updated once per image processed.
    """
    key = row["key"]
    sat_url = row["sat_url"]
    file_path = os.path.join(sat_save_dir, f"{key}.jpg")

    if resize_size != 256:
        sat_url = sat_url.replace("width=256", f"width={resize_size}")
        sat_url = sat_url.replace("height=256", f"height={resize_size}")
    
    # Check if file already exists; if so, skip the download.
    if os.path.exists(file_path):
        print(f"SKIPPED: Image for key {key} already exists.")
        with progress_lock:
            pbar.update(1)
        return None

    # Optional: use headers to mimic a browser.
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/115.0.0.0 Safari/537.36"
        )
    }
    
    max_retries = 10
    success = False
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(sat_url, headers=headers, timeout=10)
            response.raise_for_status()  # Raise an error for bad HTTP status codes.
            # Convert the image to JPEG if necessary
            image = Image.open(BytesIO(response.content))
            image = image.convert("RGB")  # Ensure compatibility with JPEG format
            image.save(file_path, "JPEG")

            # Test to see file is corrupted
            with Image.open(file_path) as img:
                img.verify()  # Does minimal decoding, good for quick validation
                success = True
                break  # Exit the loop if the download is successful.
        
        # OSError can catch issues like truncated files, permission errors, etc.
        except (UnidentifiedImageError, OSError):
            if attempt < max_retries:
                print(f"[Corrupted] Retrying: Failed attempt {attempt} for key {key} from URL {sat_url}")
                time.sleep(2)  

        except Exception as e:
            if attempt < max_retries:
                print(f"Retrying: Failed attempt {attempt} for key {key} from URL {sat_url}")
                time.sleep(2)  
            # else:
            


    if not success:
        print(f"FAILURE: Could not download image for key {key} from URL: {sat_url} after {max_retries} attempts")

    # Update the progress bar regardless of success or failure.
    with progress_lock:
        pbar.update(1)
    if not success:
        return (key, sat_url)
    return None  

def chunked_iterator(iterable, chunk_size):
    """
    Yield successive chunks of size `chunk_size` from the iterable.
    """
    iterator = iter(iterable)
    while True:
        chunk = list(itertools.islice(iterator, chunk_size))
        if not chunk:
            break
        yield chunk

# Define chunk size and number of worker threads.
chunk_size = 1000    # Process 10,000 rows at a time.
max_workers = 32     # Number of threads to use in parallel.

# Process the dataset in chunks.
try:
    for chunk in chunked_iterator(ds, chunk_size):
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit a download task for each row in the chunk.
            futures = {executor.submit(download_image, row): row for row in chunk}
            # As each future completes, record any failures.
            for future in as_completed(futures):
                result = future.result()
                if result:
                    key, sat_url = result
                    download_failures[key] = sat_url
except: 
    print(f"Download failures: {download_failures}")
    print("len(download_failures):", len(download_failures))

# Close the progress bar when done.
pbar.close()

print(f"Download failures: {download_failures}")
print("len(download_failures):", len(download_failures))
