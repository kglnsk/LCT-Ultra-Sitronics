import argparse
import time
import torch
import cv2
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine

from lightglue import LightGlue, SIFT
from lightglue.utils import load_image
from lightglue import match_pair

def read_tif_image(file_path):
    with rasterio.open(file_path) as src:
        image = src.read((1, 2, 3))
        image = image / 16384.0  # Normalize to [0, 1]
        epsg_code = src.crs.to_epsg()
        transform = src.transform
    return image, epsg_code, transform

def pixel_to_geo(transform, pixel_coords):
    geo_coords = [rasterio.transform.xy(transform, coord[1], coord[0]) for coord in pixel_coords]
    return geo_coords

def get_matching_keypoints(kp1, kp2, idxs):
    mkpts1 = kp1[idxs[:, 0]].detach().numpy()
    mkpts2 = kp2[idxs[:, 1]].detach().numpy()
    return mkpts1, mkpts2

def save_coords_to_csv(data, filename='./results/coords.csv'):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

def main(crop_name, layout_name):

    # Load images
    crop, _, _ = read_tif_image(crop_name)
    layout, epsg_code, transform = read_tif_image(layout_name)

    # Convert to torch.Tensor and match dimensions (3, H, W)
    crop = torch.tensor(crop, dtype=torch.float32)
    layout = torch.tensor(layout, dtype=torch.float32)
    
    start_time = time.strftime("%Y-%m-%dT%H:%M:%S")

    # Initialize extractor and matcher
    extractor = SIFT(max_num_keypoints=8192).eval()
    matcher = LightGlue(features='sift').eval()

    # Match keypoints
    feats0, feats1, matches01 = match_pair(extractor, matcher, crop, layout)
    match_pairs = matches01['matches']
    kps0 = feats0['keypoints']
    kps1 = feats1['keypoints']
    mkpts0, mkpts1 = get_matching_keypoints(kps0, kps1, match_pairs)
    print(match_pairs)
    
    # Find Fundamental Matrix
    Fm, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 1.5, 0.9, 100000)

    # Extract coordinates
    layout_height, layout_width = layout.shape[1:]
    corners = np.array([[0, 0], [layout_width, 0], [layout_width, layout_height], [0, layout_height]], dtype='float32')
    corners_transformed = cv2.perspectiveTransform(np.array([corners]), Fm)[0]

    # Convert pixel coordinates to geographic coordinates
    geo_coords = pixel_to_geo(transform, corners_transformed)

    # Prepare data for CSV
    data = {
        "layout_name": [layout_name],
        "crop_name": [crop_name],
        "ul": [f"{geo_coords[0][0]}; {geo_coords[0][1]}"],
        "ur": [f"{geo_coords[1][0]}; {geo_coords[1][1]}"],
        "br": [f"{geo_coords[2][0]}; {geo_coords[2][1]}"],
        "bl": [f"{geo_coords[3][0]}; {geo_coords[3][1]}"],
        "crs": [f"EPSG:{epsg_code}"],
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
