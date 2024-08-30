from flask import Flask, request, jsonify
from werkzeug.exceptions import BadRequest
from replicate.exceptions import ModelError
import os
import sys
import traceback
from dotenv import load_dotenv
import torch
from PIL import Image, ImageShow
import requests
import logging
from io import BytesIO
from openai import OpenAI
import replicate

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# for now, config is what we use for global variables
from config import *

app = Flask(__name__)
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file (if it exists)
load_dotenv()

# openai_api_key = os.getenv('OPENAI_API_KEY')

# try:
#     # Create the OpenAI client
#     client = OpenAI(api_key=openai_api_key)
# except Exception as e:
#     print_color(f"Error initializing OpenAI client: {e}", "red")
#     exit(1)


@app.route('/process_image', methods=['POST'])
def process_image():
    data = request.json
    image_url = data.get('image_url')
    detector_prompt = data.get('detector_prompt')
    without_mask_prompt = data.get('without_mask_prompt', '')
    inpainting_prompt = data.get('inpainting_prompt')
    replace_surroundings = data.get('replace_surroundings', True)

    if not image_url or not detector_prompt or not inpainting_prompt:
        return jsonify({'error': 'Missing required parameters. Please include image_url, detector_prompt, and inpainting_prompt'}), 400

    try:
        # Check if the image URL is valid
        response = requests.head(image_url)
        if response.status_code != 200:
            return jsonify({'error': f'Invalid image URL. Status code: {response.status_code}'}), 400

        # Step 1: Get mask from grounded_sam API
        logger.info("Requesting mask from grounded_sam API")
        mask_input = {
            "image": image_url,
            "mask_prompt": detector_prompt,
            "adjustment_factor": -15,
        }
        if without_mask_prompt:
            mask_input["negative_mask_prompt"] = without_mask_prompt

        try:
            mask_output = replicate.run(
                "schananas/grounded_sam:ee871c19efb1941f55f66a3d7d960428c8a5afcb77449547fe8e5a3ab9ebc21c",
                input=mask_input
            )
            mask_output_list = list(mask_output)
            logger.info(f"Mask output: {mask_output_list}")

            if len(mask_output_list) < 4:
                return jsonify({'error': 'Unexpected output from Grounded SAM model'}), 500

            mask_url = mask_output_list[2] if replace_surroundings else mask_output_list[3]

        except ModelError as e:
            logger.error(f"Grounded SAM model error: {str(e)}")
            return jsonify({'error': f'Error in Grounded SAM model: {str(e)}'}), 500

        # Step 2: Pass the mask to Stability for image inpainting
        logger.info("Requesting image inpainting from Stability API")
        try:
            inpainting_output = replicate.run(
                "stability-ai/stable-diffusion-inpainting:95b7223104132402a9ae91cc677285bc5eb997834bd2349fa486f53910fd68b3",
                input={
                    "mask": mask_url,
                    "image": image_url,
                    "prompt": inpainting_prompt,
                    "width": 448,
                    "height": 704,
                    "scheduler": "DPMSolverMultistep",
                    "num_outputs": 1,
                    "guidance_scale": 7.5,
                    "num_inference_steps": 25
                }
            )
            inpainting_output_list = list(inpainting_output)
            logger.info(f"Inpainting output: {inpainting_output_list}")

            if not inpainting_output_list:
                return jsonify({'error': 'Unexpected output from Stable Diffusion Inpainting model'}), 500

            generated_image_url = inpainting_output_list[0]

        except ModelError as e:
            logger.error(f"Stable Diffusion Inpainting model error: {str(e)}")
            return jsonify({'error': f'Error in Stable Diffusion Inpainting model: {str(e)}'}), 500

        return jsonify({
            'mask_url': mask_url,
            'generated_image_url': generated_image_url
        }), 200

    except Exception as e:
        logger.error(f"Error in process_image: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/validate', methods=['POST'])
def validate():
    '''
    This is a test endpoint to validate the JSON payload sent to the server.
    '''
    try:
        data = request.get_json()  # This will raise BadRequest if JSON is invalid
        # Process the valid JSON data
        return jsonify({"message": "JSON is valid", "data": data}), 200
    except BadRequest as e:
        # Customize the error response
        return jsonify({"error": "Invalid JSON", "message": "Failed to decode JSON object"}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))