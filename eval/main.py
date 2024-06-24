import argparse
import time
import torch
import cv2
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine

from lightglue import LightGlue, SIFT, ALIKED, SuperPoint, DISK,DoGHardNet
from lightglue.utils import load_image
from lightglue import match_pair
import numpy as np
from rasterio.enums import Resampling

import matplotlib.pyplot as plt

def normalize_sentinel2(image, lower_percent=1, upper_percent=99):
    """
    Normalize Sentinel-2 image.
    
    :param image: Input image as a numpy array
    :param lower_percent: Lower percentile for normalization (default: 2)
    :param upper_percent: Upper percentile for normalization (default: 98)
    :return: Normalized image as a numpy array (0-255, uint8)
    """
    # Convert to float for calculations
    image_float = image.astype(float)

    # Calculate percentiles
    lower_val = np.percentile(image_float, lower_percent)
    upper_val = np.percentile(image_float, upper_percent)

    # Clip the image to the calculated percentiles
    image_clipped = np.clip(image_float, lower_val, upper_val)

    # Normalize to 0-1 range
    image_normalized = (image_clipped - lower_val) / (upper_val - lower_val)

    # Scale to 0-255 and convert to uint8
    image_scaled = (image_normalized * 1.0)

    return image_scaled

# Modified load_images function
def load_images(layout_path, crop_path):
    # Load layout (geotiff)
    with rasterio.open(layout_path) as src:
        layout = load_and_normalize_band(src, (1,2,3))  # Assuming you want the first band
        transform = src.transform
        crs = src.crs
    # Load crop (regular tiff)
    with rasterio.open(crop_path) as src:
        crop = load_and_normalize_band(src, (1,2,3))  # Assuming you want the first band
    return layout, crop, transform, crs

# In your main function:
# Function to load and normalize a single band
def load_and_normalize_band(src, band_index):
    band = src.read(band_index)
    return normalize_sentinel2(band)

def pixel_to_geo(transform, pixel_coords):
    geo_coords = [rasterio.transform.xy(transform, coord[1], coord[0]) for coord in pixel_coords]
    return geo_coords

def get_matching_keypoints(kp1, kp2, idxs):
    mkpts1 = kp1[idxs[:, 0]].detach().numpy()
    mkpts2 = kp2[idxs[:, 1]].detach().numpy()
    return mkpts1, mkpts2

def save_coords_to_csv(data, filename='coords.csv'):
    df = pd.DataFrame(data)
    try:
        # Try to read existing CSV file
        existing_df = pd.read_csv(filename)
        # Append new data to existing data
        updated_df = pd.concat([existing_df, df], ignore_index=True)
    except FileNotFoundError:
        # If file doesn't exist, use the new data as is
        updated_df = df
    
    # Save the updated dataframe
    updated_df.to_csv(filename, index=False)


def main(crop_name, layout_name):

    # Load images

    layout, crop, transform, src_crs = load_images(layout_name, crop_name)

    # Convert to torch.Tensor and match dimensions (3, H, W)
    crop = torch.tensor(crop, dtype=torch.float32)
    layout = torch.tensor(layout, dtype=torch.float32)
    

    # Initialize extractor and matcher
    extractor = DoGHardNet(max_num_keypoints=8192).eval()
    matcher = LightGlue(features='doghardnet').eval()

    # Match keypoints
    start_time = time.strftime("%Y-%m-%dT%H:%M:%S")
    feats0, feats1, matches01 = match_pair(extractor, matcher, crop, layout)
    match_pairs = matches01['matches']
    kps0 = feats0['keypoints']
    kps1 = feats1['keypoints']
    mkpts0, mkpts1 = get_matching_keypoints(kps0, kps1, match_pairs)
    src_pts = mkpts0
    dst_pts = mkpts1
    
    # Find Fundamental Matrix
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.USAC_MAGSAC, 5.0)

    # Extract coordinates
    # Get crop image dimensions
    h, w = crop.shape[1:]

    # Define crop corner points
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    print(pts)
    print(H)
    # Transform crop corner points
    dst = cv2.perspectiveTransform(pts, H)
    
    # Convert pixel coordinates to geospatial coordinates
    geo_coords = []
    for point in dst.reshape(-1, 2):
        x, y = point
        lon, lat = rasterio.transform.xy(transform, y, x)
        geo_coords.append((lon, lat))
    print(geo_coords)

    data = {
        "layout_name": [layout_name],
        "crop_name": [crop_name],
        "ul": [f"{geo_coords[0][0]}; {geo_coords[0][1]}"],
        "ur": [f"{geo_coords[3][0]}; {geo_coords[3][1]}"],
        "br": [f"{geo_coords[2][0]}; {geo_coords[2][1]}"],
        "bl": [f"{geo_coords[1][0]}; {geo_coords[1][1]}"],
        "crs": [f"EPSG:{src_crs.to_epsg()}"],
        "start": [start_time],
        "end": [time.strftime("%Y-%m-%dT%H:%M:%S")]
    }

    # Save to CSV
    save_coords_to_csv(data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process crop and layout images to find matching coordinates.")
    parser.add_argument("--crop_name", type=str, required=True, help="Full path to the crop image .tif file.")
    parser.add_argument("--layout_name", type=str, required=True, help="Full path to the layout image .tif file.")
    args = parser.parse_args()

    main(args.crop_name, args.layout_name)
