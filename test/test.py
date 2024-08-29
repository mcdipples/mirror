import os
from io import BytesIO
import sys
import torch
from PIL import Image
import requests
import json
import matplotlib.pyplot as plt


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from utils.dalle import unpack_dalle_generated_images
from utils.mirror_utils import print_color

# Define the payload for the POST request
payload = {
    "image_url": "https://images.pexels.com/photos/106399/pexels-photo-106399.jpeg",
    "detector_prompt": "garage door on right side of image",
    "inpainting_prompt": "Photorealistic image of floor-to-ceiling glass doors, offering a clear view into a sleek, modern living room. The view is from outside looking in, during daytime with soft natural light. The living room features minimalist furniture, including a low-profile gray sofa, a glass coffee table, and abstract art on the walls. Warm wood flooring contrasts with white walls. Subtle reflections on the glass doors hint at the outdoor environment. The transition from exterior to interior is seamless, with the glass doors perfectly fitted into the existing house structure."
}

test_path = os.path.join(os.path.dirname(__file__), "test")
try:
    print_color("Sending POST request to the server...", "blue")
    # Send the POST request to the server
    response = requests.post("https://mirror-ovew6mtjoa-uc.a.run.app/process_image", json=payload)
    response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)

    # Check if the request was successful
    if response.status_code == 200:
        print_color("POST request successful. Processing response...", "green")
        edited_image_urls = response.json()["edited_images"]
        # edited_image_filepaths = unpack_dalle_generated_images(edited_image_urls, output_path=test_path)

        # Download and display the images
        images = []
        for url in edited_image_urls:
            try:
                print_color(f"Downloading image from URL: {url}", "blue")
                response = requests.get(url)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
                images.append(image)
                print_color(f"Image downloaded and opened successfully from URL: {url}", "green")
            except Exception as e:
                print_color(f"Error downloading or opening image from URL {url}: {e}", "red")

        # Display images in a grid
        if images:
            print_color("Displaying images in a grid...", "blue")
            fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
            for ax, img in zip(axes, images):
                ax.imshow(img)
                ax.axis('off')
            plt.show()
        else:
            print_color("No images to display.", "red")
    else:
        print_color(f"Error: {response.status_code}", "red")
        print(response.json())
except requests.exceptions.RequestException as e:
    print_color(f"Request failed: {e}", "red")
except Exception as e:
    print_color(f"An unexpected error occurred: {e}", "red")