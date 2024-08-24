from flask import Flask, request, jsonify
import os
import torch
from PIL import Image, ImageShow
import requests
from io import BytesIO

# Import necessary functions from other files
from utils.model_manager import ModelManager
from utils.mirror_utils import (
    get_device, 
    print_color,
    download_image, 
    temporary_files, 
)
from utils.image_processing import (
    to_png
)
from utils.dalle import (
    dalle_preprocess_mask, 
    dalle_inpainting, 
    resize_pil_image_to_dalle_standard_size
)

# for now, config is what we use for global variables
from config import *

app = Flask(__name__)

# Initialize Model Manager for object detection (DINO) and segmentation (SAM)
models = ModelManager(
    dino_checkpoint=DINO_CHECKPOINT_PATH, 
    dino_config = DINO_CONFIG_PATH, 
    sam_checkpoint=SAM_CHECKPOINT_PATH
)

# example of a post request using query parameters:
# curl -X POST -d "image_url=https://example.com/image.jpg&detector_prompt=detect_objects&inpainting_prompt=replace_objects" http://localhost:8080/process_image

@app.route('/process_image', methods=['POST'])
def process_image():
    data = request.json
    image_url = data['image_url']
    detector_prompt = data['detector_prompt']
    inpainting_prompt = data['inpainting_prompt']

    if not image_url or not detector_prompt or not inpainting_prompt:
        return jsonify({'error': 'Missing required parameters, please make sure to include image_url, detector_prompt, and inpainting_prompt'}), 400

    with temporary_files() as temp_dir:         
        try:
            # Download image from URL (returns PIL Image)
            image, image_format = download_image(image_url)

            # resize to fit dalle standard size
            # TODO: make preserve_aspect_ratio optional for the user, set to False for now.
            image, image_png_save_path = resize_pil_image_to_dalle_standard_size(image)

            # Detect and segment the image
            # TODO: allow for multiple masks (for multiple objects)
            binary_mask = models.detect_and_segment(image, detector_prompt, save_path=temp_dir)

            # convert binary mask to transparent mask, also saves to png
            dalle_mask_path = os.path.join(temp_dir, "mask.png")
            dalle_preprocess_mask(binary_mask, dalle_mask_path)

            # annotation example: give the mask to the model to annotate
            # annotated_image = models.annotate(image, binary_mask)
            # ImageShow.show(annotated_image)

            # inpaint the image
            edited_image_urls = dalle_inpainting(image_path=image_png_save_path, mask_path=dalle_mask_path, output_path=temp_dir, prompt=inpainting_prompt)

            # # Convert edited images to base64
            # edited_images_base64 = []
            # for path in edited_file_paths:
            #     with open(path, "rb") as img_file:
            #         encoded_string = base64.b64encode(img_file.read()).decode('utf-8')
            #         edited_images_base64.append(encoded_string)

            return jsonify({'edited_images': edited_image_urls}), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))