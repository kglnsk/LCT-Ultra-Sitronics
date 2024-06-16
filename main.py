import cv2
import numpy as np

from matching import get_matcher
from matching import get_matcher, viz2d, available_models

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse,PlainTextResponse
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
device = 'cpu'
matcher = get_matcher('doghardnet-lg', device=device)
img_size = 512

def load_image(file, resize):
    # Open the file with rasterio
    with rasterio.open(file) as src:
        # Read the first 3 channels
        img = src.read([1, 2, 3])
        # Resize image (this is a placeholder, actual resizing needs to be implemented)
        # For simplicity, we assume the image is already in the required size.
    return img

@app.post("/image-match", response_class=PlainTextResponse)
async def image_match(img0_file: UploadFile = File(...), img1_file: UploadFile = File(...)):
    # Load images
    img0 = load_image(img0_file.file, resize=img_size)
    img1 = load_image(img1_file.file, resize=img_size * 6)
    
    # Perform image matching
    result = matcher(img0, img1)
    num_inliers, H, mkpts0, mkpts1 = result['num_inliers'], result['H'], result['mkpts0'], result['mkpts1']
    
    # Get coordinates of the four corners
    def get_corner_coordinates(img, H, crs):
        # Get image dimensions
        h, w = img.shape[1:]
        corners = np.array([
            [0, 0],        # Top-left
            [w, 0],        # Top-right
            [w, h],        # Bottom-right
            [0, h]         # Bottom-left
        ])
        # Apply homography
        corners_transformed = cv2.perspectiveTransform(np.float32([corners]), H)[0]
        
        # Convert pixel coordinates to world coordinates using gdal
        src_ds = gdal.Open(img0_file.file.name)
        src_proj = src_ds.GetProjection()
        src_geotrans = src_ds.GetGeoTransform()
        
        dst_proj = crs.ExportToWkt()
        dst_geotrans = src_geotrans
        
        transform = gdal.Transformer(src_ds, None, [])
        
        world_coords = []
        for corner in corners_transformed:
            px, py = corner
            success, (world_x, world_y, _) = transform.TransformPoint(False, px, py)
            if success:
                world_coords.append((world_x, world_y))
        
        return world_coords
    
    # Get EPSG:32637 CRS
    crs = gdal.osr.SpatialReference()
    crs.ImportFromEPSG(32637)
    
    # Transform coordinates
    corner_coords = get_corner_coordinates(img0, H, crs)
    
    # Format string output
    formatted_coords = "\n".join([f"{x:.3f};{y:.3f}" for x, y in corner_coords])
    
    return formatted_coords


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
