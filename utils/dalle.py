from PIL import Image
import numpy as np
import os
import traceback
from dotenv import load_dotenv
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
    try:
        print_color("Resizing image to DALL-E standard size...", "blue")
        dalle_standard_sizes = [1024, 512, 256]
        
        # Find the largest size that both dimensions of the image are greater than or equal to
        max_dimension = max(image.width, image.height)
        target_size = next((size for size in dalle_standard_sizes if max_dimension >= size), 256)
        
        resized_image = image.resize((target_size, target_size), Image.LANCZOS)
        print_color(f"Image resized to {target_size}x{target_size}", "green")
        return resized_image
    except Exception as e:
        print_color(f"Error resizing image: {e}", "red")
        return image

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
    try:
        print_color("Preprocessing mask for DALL-E...", "blue")
        transparent_mask = convert_binary_to_transparent_mask(binary_mask)
        to_png(transparent_mask, mask_png_save_path)
        print_color(f"Mask saved to {mask_png_save_path}", "green")
    except Exception as e:
        print_color(f"Error preprocessing mask: {e}", "red")

def dalle_preprocess_image(image: Image.Image, image_png_save_path: str) -> None:
    """
    Preprocess the image for DALL-E.
    """
    try:
        print_color("Preprocessing image for DALL-E...", "blue")
        image = resize_pil_image_to_dalle_standard_size(image)
        to_png(image, image_png_save_path)
        print_color(f"Image saved to {image_png_save_path}", "green")
        return image
    except Exception as e:
        print_color(f"Error preprocessing image: {e}", "red")
        return None, None

def unpack_dalle_generated_images(response):
    """
    Process the images from the DALL-E 3 response.

    Args:
        response: The response from the DALL-E 2 API.
        filepath (str): The filepath to save the images.

    Returns:
        list[str]: The URLs of the images.
    """
    try:
        print_color("Unpacking DALL-E generated images to URL...", "blue")
        urls = [datum.url for datum in response.data]  # extract URLs
        print_color("Images unpacked to URL", "green")
        return urls
    except Exception as e:
        print_color(f"Error unpacking DALL-E generated images for URL: {e}", "red")
        return []

def unpack_and_save_dalle_generated_images(urls: list[str], filepath: str):
    """
    Process the images from the DALL-E 3 response.

    Args:
        urls (list[str]): The URLs of the images to download.
        filepath (str): The filepath to save the images.

    Returns:
        list[str]: The filepaths to the images.
    """
    try:
        print_color("Unpacking DALL-E generated images...", "blue")
        images = [requests.get(url).content for url in urls]  # download images
        image_names = [f"edited_{i + 1}.png" for i in range(len(images))]  # create names
        filepaths = [os.path.join(filepath, name) for name in image_names]  # create filepaths
        for image, filepath in zip(images, filepaths):  # loop through the variations
            with open(filepath, "wb") as image_file:  # open the file
                image_file.write(image)  # write the image to the file
        print_color("Images unpacked and saved successfully.", "green")
        return filepaths
    except Exception as e:
        print_color(f"Error unpacking DALL-E generated images: {e}", "red")
        return []

def dalle_inpainting(image_path: str, mask_path: str, output_path: str, prompt: str,
size: tuple = (1024, 1024), client: OpenAI = None):
    """
    Inpaint the image using the DALL-E 2 API.

    Args:
        image_path (str): The path to the image to edit.
        mask_path (str): The path to the mask to use for the edit.
        output_path (str): The path to save the edited images.
        prompt (str): The prompt to use for the edit.
        size (str, optional): The size of the image to generate. Default is "1024x1024".
        client (OpenAI, optional): The OpenAI client to use for the edit. Default is None.

    Returns:
        edit_response (dict): The response from the DALL-E 2 API. 
    """
    try:
        print_color("Initializing OpenAI client...", "blue")
        # print_color(f"Client API key: {os.environ['OPENAI_API_KEY']}...", "grey")
        print(f"Client API key: {client.api_key[:5]}...")
        print(f"Image path: {image_path}")
        print(f"Mask path: {mask_path}")
        print(f"Prompt: {prompt}")
        print(f"Size: {size}")    
    except Exception as e:
        print_color(f"Error initializing OpenAI client: {e}", "red")
        return None

    try:
        print_color("Editing image with DALL-E...", "blue")
        edit_response = client.images.edit(
            image=open(image_path, "rb"),
            mask=open(mask_path, "rb"),
            prompt=prompt,
            n=3,
            size=f"{size[0]}x{size[1]}",
            response_format="url",
        )
        print_color("Image edited successfully.", "green")
    except Exception as e:
        print(f"Error during DALL-E inpainting: {str(e)}")
        print(f"Error type: {type(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return None

    return edit_response