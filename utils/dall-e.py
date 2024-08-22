from PIL import Image
import numpy as np
import os
import requests

from utils.image_processing import resize_pil_image_to_dalle_standard_size, to_png

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
    transparent_mask = convert_binary_to_transparent_mask(binary_mask, image.size)
    to_png(transparent_mask, mask_png_save_path)

def dalle_postprocess(masks: np.ndarray) -> np.ndarray:
    """
    Postprocess the masks for DALL-E.
    """
    return masks

def resize_pil_image_to_dalle_standard_size(image: Image.Image, preserve_aspect_ratio: bool = True) -> Image.Image:
    """
    Resize the PIL image to 1024x1024, 512x512, or 256x256 based on the image's max size.

    Args:
        image (Image.Image): The image to resize.
        preserve_aspect_ratio (bool, optional): Whether to preserve the aspect ratio. Default is True.

    Returns:
        Image.Image: Resized image.
    """
    max_size = max(image.size)
    if max_size > 1024:
        new_size = (1024, 1024)
    elif max_size > 512:
        new_size = (512, 512)
    else:
        new_size = (256, 256)

    return resize_pil_image(
        image, 
        height=new_size[1], 
        width=new_size[0], 
        preserve_aspect_ratio=preserve_aspect_ratio
    )

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

    try: # return filepaths to the edited images
        edited_files = unpack_dalle_generated_images(edit_response, output_path)
    except Exception as e:
        print(f"\033[91mError unpacking images: {e}\033[0m")
        return None

    return edited_files