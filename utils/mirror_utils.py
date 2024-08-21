import os
import torch
import cv2
import numpy as np
import streamlit as st
import requests
from PIL import Image

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from groundingdino.util.inference import Model


from config import *
# ============================================================
# Helper Functions
# ============================================================

def print_color(text, color):
    colors = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'purple': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m',
    }
    end_color = '\033[0m'
    print(f"{colors[color]}{text}{end_color}")

# ------------------------------------------------------------
# '''
# FUNCTION: get_device
# This function checks if CUDA devices are available. If a CUDA device is available, it returns the CUDA device.
# Otherwise, it returns the CPU device.
# '''
# ------------------------------------------------------------
def get_device() -> str:
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"\033[1;32m📀CUDA DEVICE: {device_name}\033[0m")
        # st.info(f"CUDA EXECUTABLE LOCATION: {torch.cuda.get_executable_path()}")
        return torch.device("cuda")
    else:
        print("\033[93mNo CUDA devices available. Using CPU.\033[0m")
        return torch.device("cpu")
# ------------------------------------------------------------


# ------------------------------------------------------------
# '''
# FUNCTION: to_png
# This function converts an image to a PNG file.
# INPUT: image: a numpy array or PIL Image
# RETURNS the path to the PNG file.
# '''
# ------------------------------------------------------------
def to_png(image):
    if isinstance(image, np.ndarray):
        # Convert cv2 BGR image to RGB
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Create a PIL Image from the numpy array
        pil_image = Image.fromarray(image)
    elif isinstance(image, Image.Image):
        pil_image = image
    else:
        raise ValueError("Unsupported image type. Please provide a cv2 or PIL image.")

    # Save the PIL Image as PNG
    pil_image.save("image.png", format="PNG")
    return "image.png"
# ------------------------------------------------------------

# ------------------------------------------------------------
# '''
# FUNCTION: process_dalle_images
# This function processes the images from the DALL-E 3 response.
# INPUTS:
# response: the response from the DALL-E 2 API
# filename: the filename to use for the images
# image_dir: the directory to save the images
# RETURNS the filepaths to the images.
# '''
# ------------------------------------------------------------
def process_dalle_images(response, filename, image_dir):
    # save the images
    urls = [datum.url for datum in response.data]  # extract URLs
    images = [requests.get(url).content for url in urls]  # download images
    image_names = [f"{filename}_{i + 1}.png" for i in range(len(images))]  # create names
    filepaths = [os.path.join(image_dir, name) for name in image_names]  # create filepaths
    for image, filepath in zip(images, filepaths):  # loop through the variations
        with open(filepath, "wb") as image_file:  # open the file
            image_file.write(image)  # write the image to the file

    return filepaths
# ------------------------------------------------------------

# ------------------------------------------------------------
# '''
# FUNCTION: tile_image
# This function tiles the image based on the bounding boxes to a size
# that DALL-E 2 can work with. Saves these tiles to a directory and
# returns the filepaths to the tiles.
# INPUTS:
# image_path: the path to the image
# boxes: the bounding boxes of detected objects
# edit_path: the path to the directory to save the tiles
# RETURNS the filepaths to the tiles.
# '''
# dead for rn.
# ------------------------------------------------------------
# def tile_image(image_path: str, boxes, edit_path: str) -> dict[int, str]:
#     im_rgb = cv2_to_rgb(image_path)

#     print(f'IMAGE SIZE: {im_rgb.shape}')

#     tiles = {}

#     for i, box in enumerate(boxes):
#         x1, y1, x2, y2 = box
#         print(f"BOX {i} : {x1, y1, x2, y2}")

#         center_x_pix = (x1 + x2) // 2
#         center_y_pix = (y1 + y2) // 2
#         w_pix = x2 - x1
#         h_pix = y2 - y1

#         print(f"PIXELS: {center_x_pix, center_y_pix, w_pix, h_pix}")

#         print("WIDTH", w_pix)
#         print("HEIGHT", h_pix)
#         print(f'BOX COORDS: {center_x_pix, center_y_pix}')

#         if max(w_pix, h_pix) <= 256:
#             tile_size = 256
#         elif max(w_pix, h_pix) <= 512:
#             tile_size = 512
#         else:
#             tile_size = 1024

#         tile_w_crop_l = center_x_pix - (tile_size//2)
#         tile_w_crop_r = center_x_pix + (tile_size//2)
#         tile_h_crop_b = center_y_pix - (tile_size//2)
#         tile_h_crop_t = center_y_pix + (tile_size//2)

#         print(f"TILE CROP: {tile_w_crop_l} to {tile_w_crop_r}, {tile_h_crop_b} to {tile_h_crop_t}")

#         tile = im_rgb[tile_h_crop_b:tile_h_crop_t, tile_w_crop_l:tile_w_crop_r]

#         if tile.shape[0] and tile.shape[1] > 0:
#           tile = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
#           tile_path = os.path.join(edit_path, f"tile_{i}.png")
#           cv2.imwrite(tile_path, tile)

#           tiles[i] = tile_path
#           print(f"TILE {i} SAVED")
#         else:
#           print("NO TILE")
#           continue

#     return tiles
# # ------------------------------------------------------------

# ------------------------------------------------------------
# '''
# FUNCTION: mask_tiles
# This function masks the tiles using the SAM model.
# INPUTS:
# tiles: the tiles to mask
# sam_predictor: the SAM predictor
# RETURNS the masked tiles.
# '''
# ------------------------------------------------------------
def mask_tiles(tiles: list[np.ndarray], sam_predictor: SamPredictor) -> dict[str, np.ndarray]:
    sam_results = {}

    for i, tile in enumerate(tiles):
      sam_predictor.set_image(tile)

      input_point = np.array([[tile.shape[0]//2, tile.shape[1]//2]])
      input_label = np.array([1])

      masks, scores, logits = sam_predictor.predict(
          point_coords=input_point,
          point_labels=input_label,
          multimask_output=True,)
      print(f'Tile {i} : Masks Shape: {masks.shape}')
      sam_results[f'Tile {i}'] = masks

    return sam_results


