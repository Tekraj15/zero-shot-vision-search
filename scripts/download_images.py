import os
import pandas as pd
import requests
from tqdm import tqdm
import argparse

def download_images(csv_path, output_dir, limit=10000):
    """
    Download images from Unsplash Lite dataset CSV.
    
    Args:
        csv_path (str): Path to the photos.csv000 file.
        output_dir (str): Directory to save images.
        limit (int): Maximum number of images to download.
    """
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    os.makedirs(output_dir, exist_ok=True)

    print(f"Reading {csv_path}...")
    try:
        # Read TSV file
        df = pd.read_csv(csv_path, sep='\t')
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    print(f"Found {len(df)} images in dataset. Downloading top {limit}...")

    success_count = 0
    
    # Iterate through the dataframe
    for index, row in tqdm(df.iterrows(), total=min(len(df), limit)):
        if success_count >= limit:
            break
            
        photo_id = row['photo_id']
        photo_url = row['photo_image_url']
        
        # Construct output path
        # Using .jpg as default extension, though some might be png. 
        # Unsplash usually serves jpg.
        output_path = os.path.join(output_dir, f"{photo_id}.jpg")
        
        if os.path.exists(output_path):
            # print(f"Skipping {photo_id}, already exists.")
            success_count += 1
            continue

        try:
            response = requests.get(photo_url, timeout=10)
            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                success_count += 1
            else:
                pass
                # print(f"Failed to download {photo_id}: Status {response.status_code}")
        except Exception as e:
            pass
            # print(f"Error downloading {photo_id}: {e}")

    print(f"Download complete. Successfully downloaded {success_count} images to {output_dir}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download images from Unsplash Lite dataset.")
    parser.add_argument("--csv", type=str, default="assets/unsplash-research-dataset-lite-latest/photos.csv000", help="Path to the photos CSV file.")
    parser.add_argument("--output", type=str, default="assets/image-dataset", help="Directory to save images.")
    parser.add_argument("--limit", type=int, default=50, help="Number of images to download.")
    
    args = parser.parse_args()
    
    download_images(args.csv, args.output, args.limit)
