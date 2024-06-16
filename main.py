import cv2
import numpy as np

from matching import get_matcher
from matching import get_matcher, viz2d, available_models

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import rasterio


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = 'cpu' # 'cpu'
matcher = get_matcher('doghardnet-lg', device=device)  # Choose any of our ~20 matchers listed below
img_size = 512

def match_image(image):
    return 0    

def adaptive_thresholds(image, underexposed_factor=0.15, overexposed_factor=5.0):
    """
    Computes adaptive thresholds for underexposed and overexposed pixels based on the median value of each channel.

    Parameters:
    - image: Input image (numpy array).
    - underexposed_factor: Factor for determining underexposed pixel threshold (15% of median).
    - overexposed_factor: Factor for determining overexposed pixel threshold (500% of median).

    Returns:
    - lower_thresh: Adaptive lower threshold for each channel.
    - upper_thresh: Adaptive upper threshold for each channel.
    """
    lower_thresh = []
    upper_thresh = []

    # Iterate over each channel
    for i in range(image.shape[2]):
        channel = image[:, :, i]
        median_val = np.median(channel)

        lower_thresh.append(median_val * underexposed_factor)
        upper_thresh.append(median_val * overexposed_factor)
    
    return lower_thresh, upper_thresh

def detect_and_fix_dead_pixels_multispectral(image, filter_size=3, underexposed_factor=0.15, overexposed_factor=5.0):
    """
    Detects and fixes dead pixels in a multispectral image using a median filter.

    Parameters:
    - image: Input image with multiple channels (numpy array).
    - filter_size: Size of the median filter.
    - underexposed_factor: Factor for determining underexposed pixel threshold.
    - overexposed_factor: Factor for determining overexposed pixel threshold.

    Returns:
    - Fixed image (numpy array).
    """
    fixed_image = np.copy(image)
    lower_thresh, upper_thresh = adaptive_thresholds(image, underexposed_factor, overexposed_factor)

    # Apply median filter to each channel
    for i in range(image.shape[2]):
        channel = image[:, :, i]
        mask = (channel == 0) | (channel < lower_thresh[i]) | (channel > upper_thresh[i])
        median_filtered_channel = cv2.medianBlur(channel, filter_size)
        fixed_image[:, :, i][mask] = median_filtered_channel[mask]

    return fixed_image


def find_difference(original_image,fixed_image):
# Find the indices where the images differ
    differences = np.where(original_image != fixed_image)

    # Unpack the indices into row, column, and channel arrays
    rows, cols, channels = differences

    # Create a list to store the report lines
    report_lines = []

    # Iterate over the indices of the differing pixels
    for i in range(len(rows)):
        row = rows[i]
        col = cols[i]
        channel = channels[i]
        original_value = original_image[row, col, channel]
        fixed_value = fixed_image[row, col, channel]

        # Append the report line to the list
        report_lines.append(f"{row}; {col}; {channel + 1}; {original_value}; {fixed_value}")

    # Join the report lines into a single string
    report_string = "\n".join(report_lines)

    # Create a JSON object with a single field "report"
    report_json = {"report": report_string}
    return report_json





app.post("/pixels/")
async def upload_tif(file: UploadFile = File(...)):
    # Check if the uploaded file is a TIFF image
    if file.content_type != "image/tif":
        raise HTTPException(status_code=400, detail="Invalid file type. Only TIFF images are accepted.")

    try:
        print('Here',flush = True)
        # Read the uploaded file
        contents = await file.read()
        # Create a PIL Image object
        byte_stream = io.BytesIO(contents)

        # Open the byte stream with rasterio
        with rasterio.open(byte_stream) as src:
            crop = src.read().transpose(1,2,0)
        
        fixed_image = detect_and_fix_dead_pixels_multispectral(crop)
        report = find_difference(crop,fixed_image)
        
        # Check if the image is actually a TIFF
        return JSONResponse(content = report)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the image: {str(e)}")
