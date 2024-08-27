import os
import torch
import cv2
import numpy as np
import requests
import tempfile
import shutil
from PIL import Image
from contextlib import contextmanager   
from io import BytesIO
from config import *
# ============================================================
# Helper Functions
# ============================================================

def print_color(text, color):
    """
    Print text in a specified color.

    Args:
        text (str): The text to print.
        color (str): The color to use for the text. Available options are:
            'red', 'green', 'yellow', 'blue', 'purple', 'cyan', 'white', 'black', 'orange', 'pink', 'gray'.
    """
    colors = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'purple': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m',
        'black': '\033[90m',
        'orange': '\033[38;5;208m',
        'pink': '\033[38;5;205m',
        'gray': '\033[37m',
    }
    end_color = '\033[0m'
    print(f"{colors[color]}{text}{end_color}")

# def download_image(url: str, save_path: str) -> str:

def get_device() -> str:
    """
    Check if CUDA devices are available. If a CUDA device is available, return the CUDA device.
    Otherwise, return the CPU device.

    Returns:
        str: The device to use ('cuda' or 'cpu').
    """
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"\033[1;32m📀CUDA DEVICE: {device_name}\033[0m")
        # st.info(f"CUDA EXECUTABLE LOCATION: {torch.cuda.get_executable_path()}")
        return torch.device("cuda")
    else:
        print("\033[93mNo CUDA devices available. Using CPU.\033[0m")
        return torch.device("cpu")

@contextmanager
def temporary_files():
    """Context manager for creating and cleaning up temporary files."""
    temp_dir = tempfile.mkdtemp()
    try:
        yield temp_dir
    finally:
        # Clean up the temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)

def download_image(url: str) -> tuple[Image.Image, str]:
    """
    Download an image from a URL.

    Args:
        url (str): The URL of the image to download.

    Returns:
        tuple: A tuple containing the downloaded image (Image.Image) and the original image file type (str).
    """
    try:
        response = requests.get(url, timeout=4.0)
        image = Image.open(BytesIO(response.content))
        image_format = image.format

        return image, image_format
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Failed to download image: {str(e)}")
    except IOError as e:
        raise ValueError(f"Failed to open image: {str(e)}")

