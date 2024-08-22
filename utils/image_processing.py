from typing import Union
import numpy as np
from PIL import Image
import cv2


# ============ IMAGE PROCESSING ============

def to_png(image: Union[np.ndarray, Image.Image], filepath: str) -> str:
    """
    Convert an image to a PNG file.

    Args:
        image (Union[np.ndarray, PIL.Image.Image]): The image to convert.
        filename (str): The filename to use for the image.
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
    pil_image.save(filepath, format="PNG")
    return filepath

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
    mask[mask != 0] = 255
    mask[mask == 0] = 1
    mask[mask == 255] = 0
    mask[mask == 1] = 255    

    # Create a new transparent image
    width, height = image_size
    transparent_mask = Image.new("RGBA", (width, height), (0, 0, 0, 1))
    
    # Apply the mask to the alpha channel
    mask_data = np.array(transparent_mask)
    mask_data[:, :, 3] = mask
    
    new_mask = Image.fromarray(mask_data, "RGBA")

    return new_mask

def resize_pil_image(image: Image.Image, height: int = None, width: int = None, preserve_aspect_ratio: bool = True) -> Image.Image:
    """
    Resize the PIL image to the given height or width while maintaining the aspect ratio if specified.

    Args:
        image (Image.Image): The image to resize.
        height (int, optional): The target height for the image.
        width (int, optional): The target width for the image.
        preserve_aspect_ratio (bool, optional): Whether to preserve the aspect ratio. Default is True.

    Returns:
        Image.Image: Resized image.
    """
    if height is None and width is None:
        raise ValueError("Either height or width must be provided.")

    original_width, original_height = image.size
    aspect_ratio = original_width / original_height

    if preserve_aspect_ratio:
        if height is not None and width is not None:
            new_size = (width, height)
        elif height is not None:
            new_size = (int(height * aspect_ratio), height)
        else:
            new_size = (width, int(width / aspect_ratio))
    else:
        new_size = (width if width is not None else original_width, height if height is not None else original_height)

    resized_image = image.resize(new_size, Image.ANTIALIAS)

    return resized_image

''' 
TODO: tile_image and then mask the tiles if user wants multiple objects edited at once
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
'''
# ============ END IMAGE PROCESSING ============