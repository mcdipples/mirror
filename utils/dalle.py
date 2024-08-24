from PIL import Image
import numpy as np
import os
import requests

from utils.image_processing import (
    to_png,
    convert_binary_to_transparent_mask,
    resize_pil_image
)
from utils.mirror_utils import print_color

from openai import OpenAI

def resize_pil_image_to_dalle_standard_size(image: Image.Image, preserve_aspect_ratio: bool = True) -> Image.Image:
    """
    Resize the PIL image to 1024x1024, 512x512, or 256x256 based on the image's max size.

    Args:
        image (Image.Image): The image to resize.
        preserve_aspect_ratio (bool, optional): Whether to preserve the aspect ratio. Default is True.

    Returns:
        Image.Image: Resized image.
    """
    dalle_standard_sizes = [1024, 512, 256]
    
    # Find the largest size that both dimensions of the image are greater than or equal to
    max_dimension = max(image.width, image.height)
    target_size = next((size for size in dalle_standard_sizes if max_dimension >= size), 256)
    
    return image.resize((target_size, target_size), Image.LANCZOS)

def dalle_preprocess_mask( 
    binary_mask: np.ndarray,
    mask_png_save_path: str, 
    preserve_aspect_ratio: bool = False
) -> None:
    """
    Preprocess the masks for DALL-E.

    Args:
        binary_mask (np.ndarray): The binary mask to convert to transparent mask.
        mask_png_save_path (str): The path to save the transparent mask.
        preserve_aspect_ratio (bool, optional): Whether to preserve the aspect ratio. Default is False.

    Returns:
        None
    """
    # resize_pil_image_to_dalle_standard_size
    print_color(f"Saving mask to... {mask_png_save_path}", "blue")
    transparent_mask = convert_binary_to_transparent_mask(binary_mask)
    to_png(transparent_mask, mask_png_save_path)

def dalle_preprocess_image(image: Image.Image, image_png_save_path: str) -> None:
    """
    Preprocess the image for DALL-E.
    """
    image = resize_pil_image_to_dalle_standard_size(image)
    print_color(f"Saving image to... {image_png_save_path}", "blue")
    to_png(image, image_png_save_path)
    return image, image_png_save_path


def unpack_dalle_generated_images(response, filepath: str):
    """
    Process the images from the DALL-E 3 response.

    Args:
        response: The response from the DALL-E 2 API.
        filepath (str): The filepath to save the images.

    Returns:
        list[str]: The filepaths to the images.
    """
    # save the images
    urls = [datum.url for datum in response.data]  # extract URLs
    images = [requests.get(url).content for url in urls]  # download images
    image_names = [f"{filename}_{i + 1}.png" for i in range(len(images))]  # create names
    filepaths = [os.path.join(filepath, name) for name in image_names]  # create filepaths
    for image, filepath in zip(images, filepaths):  # loop through the variations
        with open(filepath, "wb") as image_file:  # open the file
            image_file.write(image)  # write the image to the file

    return filepaths

def dalle_inpainting(image_path: str, mask_path: str, output_path: str, prompt: str,
size: str = "1024x1024", openai_api_key=None):
    """
    Inpaint the image using the DALL-E 2 API.

    Args:
        image_path (str): The path to the image to edit.
        mask_path (str): The path to the mask to use for the edit.
        output_path (str): The path to save the edited images.
        prompt (str): The prompt to use for the edit.
        size (str, optional): The size of the image to generate. Default is "1024x1024".
        openai_api_key (str, optional): The API key to use for the edit. Default is None.

    Returns:
        edit_response (dict): The response from the DALL-E 2 API. 
    """
    try:
        openai_client = OpenAI(api_key=openai_api_key)
    except Exception as e:
        print(f"\033[91mError initializing OpenAI client: {e}\033[0m")
        return None

    try:
        print(f"\033[95mEditing image...\033[0m")
        edit_response = openai_client.images.edit(
            image=open(image_path, "rb"),
            mask=open(mask_path, "rb"),
            prompt=prompt,
            n=3,
            size=size,
            response_format="url",
        )
    except Exception as e:
        print(f"\033[91mError editing image: {e}\033[0m")
        return None

    # try: # return filepaths to the edited images
    #     edited_file_paths = unpack_dalle_generated_images(edit_response, output_path)
    # except Exception as e:
    #     print(f"\033[91mError unpacking images: {e}\033[0m")
    #     return None

    return edit_response