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
    """
    Print text in a specified color.

    Args:
        text (str): The text to print.
        color (str): The color to use for the text. Available options are:
            'red', 'green', 'yellow', 'blue', 'purple', 'cyan', 'white'.
    """
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
    temp_dir = tempfile.mkdtemp(dir='/tmp')
    try:
        yield temp_dir
    finally:
        for root, dirs, files in os.walk(temp_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(temp_dir)

def download_image(url: str) -> Image.Image:
    """
    Download an image from a URL.

    Args:
        url (str): The URL of the image to download.

    Returns:
        Image.Image: The downloaded image.
    """
    response = requests.get(url, timeout=4.0)
    image = Image.open(BytesIO(response.content))

    if response.status_code != requests.codes.ok:
        assert False, 'Status code error: {}.'.format(response.status_code)

    return image

# ============ IMAGE PROCESSING ============

def to_png(image):
    """
    Convert an image to a PNG file.

    Args:
        image (Union[np.ndarray, PIL.Image.Image]): The image to convert.

    Returns:
        str: The path to the PNG file.
    """
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


def to_bgr(img_array: np.ndarray) -> np.ndarray:
    """
    Convert the image array to BGR format if it's in RGB format.

    Args:
        img_array (np.ndarray): The image array to convert.

    Returns:
        np.ndarray: The converted image array (BGR format).
    """
    if img_array.shape[2] == 3:  # If it's a 3-channel image
        return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    elif img_array.shape[2] == 4:  # If it has an alpha channel
        return cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
    return img_array  # Return original array if it doesn't match these conditions

# def pil_to_rgb(image: Image.Image) -> np.ndarray:
#     """
#     Convert a PIL image to an RGB numpy array.
#     """
#     return np.array(image.convert("RGB"))

def process_dalle_images(response, filename, image_dir):
    """
    Process the images from the DALL-E 3 response.

    Args:
        response: The response from the DALL-E 2 API.
        filename (str): The filename to use for the images.
        image_dir (str): The directory to save the images.

    Returns:
        list[str]: The filepaths to the images.
    """
    # save the images
    urls = [datum.url for datum in response.data]  # extract URLs
    images = [requests.get(url).content for url in urls]  # download images
    image_names = [f"{filename}_{i + 1}.png" for i in range(len(images))]  # create names
    filepaths = [os.path.join(image_dir, name) for name in image_names]  # create filepaths
    for image, filepath in zip(images, filepaths):  # loop through the variations
        with open(filepath, "wb") as image_file:  # open the file
            image_file.write(image)  # write the image to the file

    return filepaths


def resize_image(image: np.ndarray, size: tuple(int, int) = None, **kwargs) -> np.ndarray:
    """
    Resize the image to the given size. Size can be `min_size` (scalar) or `(height, width)` tuple. If size is an
    int, smaller edge of the image will be matched to this number.

    Args:
        image (np.ndarray):
            The image to resize.
        size (Union[Dict[str, int], Tuple[int, int]]):
            The target size for the image. Available options are:
                - {"height": int, "width": int}: Resize the image to the exact size (height, width) without keeping the aspect ratio.
                - {"max_height": int, "max_width": int}: Resize the image to the maximum size while respecting the aspect ratio, ensuring the height is less than or equal to max_height and the width is less than or equal to max_width.
        **kwargs:
            data_format (str, optional):
                The channel dimension format for the output image. If not provided, the channel dimension format of the input image is used.
            input_data_format (str, optional):
                The channel dimension format of the input image. If not provided, it will be inferred.

    Returns:
        np.ndarray: Resized image.
    """
    data_format = kwargs.get('data_format')
    input_data_format = kwargs.get('input_data_format')

    # Determine the input image size
    if input_data_format is None:
        input_data_format = 'channels_last' if image.shape[-1] in [1, 3] else 'channels_first'
    image_size = image.shape[:2] if input_data_format == 'channels_last' else image.shape[1:]

    # Determine the new size based on the provided keyword arguments.
    if "height" in kwargs and "width" in kwargs:
        new_size = (kwargs["height"], kwargs["width"])
    elif "max_height" in kwargs and "max_width" in kwargs:
        aspect_ratio = image_size[1] / image_size[0]
        if image_size[0] > kwargs["max_height"]:
            new_size = (kwargs["max_height"], int(kwargs["max_height"] * aspect_ratio))
        elif image_size[1] > kwargs["max_width"]:
            new_size = (int(kwargs["max_width"] / aspect_ratio), kwargs["max_width"])
        else:
            new_size = image_size
    else:
        raise ValueError("Invalid size specification. Please provide either 'height' and 'width','max_height' and 'max_width'.")

    # Perform the resizing
    resized_image = cv2.resize(image, (new_size[1], new_size[0]), interpolation=cv2.INTER_LINEAR)

    return resized_image


def mask_tiles(tiles: list[np.ndarray], sam_predictor: SamPredictor) -> dict[str, np.ndarray]:
    """
    Mask the tiles using the SAM model.

    Args:
        tiles (list[np.ndarray]): The tiles to mask.
        sam_predictor (SamPredictor): The SAM predictor.

    Returns:
        dict[str, np.ndarray]: The masked tiles.
    """
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

def convert_binary_to_transparent_mask(binary_mask, image_size):
    """
    Convert a binary mask to a transparent mask.
    
    Args:
    binary_mask (numpy.ndarray): Binary mask array.
    image_size (tuple): Size of the original image (width, height).
    
    Returns:
    PIL.Image: Transparent mask as a PIL Image.
    """
    # Invert and prepare the binary mask
    mask = binary_mask.astype("uint8")
    mask = np.logical_not(mask).astype("uint8") * 255
    
    # Create a new transparent image
    width, height = image_size
    transparent_mask = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    
    # Apply the mask to the alpha channel
    mask_data = np.array(transparent_mask)
    mask_data[:, :, 3] = mask
    
    return Image.fromarray(mask_data, "RGBA")

# ============ END IMAGE PROCESSING ============

