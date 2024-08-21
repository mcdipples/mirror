from flask import Flask, request, jsonify
import os
import torch
from PIL import Image
import requests
from io import BytesIO

# Import necessary functions from other files
from utils.model_manager import ModelManager, convert_to_bgr
from utils.mirror_utils import get_device, to_png, process_dalle_images
from config import *

app = Flask(__name__)

# Initialize models
device = get_device()
dino_detector = ModelManager.load_DINO_model(device)
mask_generator, sam_predictor = ModelManager.load_SAM_model(device)

# example of a post request using query parameters:
# curl -X POST -d "image_url=https://example.com/image.jpg&detector_prompt=detect_objects&inplacing_prompt=replace_objects" http://localhost:8080/process_image

@app.route('/process_image', methods=['POST'])
def process_image():
    data = request.json
    image_url = data.get('image_url')
    detector_prompt = data.get('detector_prompt')
    inplacing_prompt = data.get('inplacing_prompt')

    if not image_url or not detector_prompt or not inplacing_prompt:
        return jsonify({'error': 'Missing required parameters'}), 400

    try:
        # Download image from URL
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))

        # Detect objects
        detections, labels = ModelManager.detect_objects(
            detector_prompt=detector_prompt,
            image=image,
            dino=dino_detector
        )

        # Convert PIL Image to numpy array and then to BGR
        img_array = convert_to_bgr(np.array(image))

        # Generate mask
        detections.mask = ModelManager.segment(
            sam_predictor=sam_predictor,
            image=cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy
        )

        # Process mask for editing
        chosen_mask = detections.mask[0]
        chosen_mask = chosen_mask.astype("uint8")
        chosen_mask[chosen_mask != 0] = 255
        chosen_mask[chosen_mask == 0] = 1
        chosen_mask[chosen_mask == 255] = 0
        chosen_mask[chosen_mask == 1] = 255

        width, height = image.size
        mask = Image.new("RGBA", (width, height), (0, 0, 0, 1))
        pix = np.array(mask)
        pix[:, :, 3] = chosen_mask
        new_mask = Image.fromarray(pix, "RGBA")

        # Save the mask temporarily
        mask_path = "temp_mask.png"
        new_mask.save(mask_path)

        # Perform image editing (you'll need to implement this part)
        # This is where you'd use the OpenAI API to edit the image
        # For now, we'll just return a placeholder URL
        edited_image_url = "https://placeholder.com/edited_image.jpg"

        # Clean up temporary files
        os.remove(mask_path)

        return jsonify({'edited_image_url': edited_image_url}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
