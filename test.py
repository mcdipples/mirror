# from flask import Flask, request, jsonify
import os
import sys
import torch
from PIL import Image
import requests

# Import necessary functions from other files
# from utils.model_manager import ModelManager, convert_to_bgr
# from utils.mirror_utils import get_device, to_png, process_dalle_images
from config import *

# app = Flask(__name__)

# # Initialize models
# device = get_device()
# dino_detector = ModelManager.load_DINO_model(device)
# mask_generator, sam_predictor = ModelManager.load_SAM_model(device)


from transformers import AutoProcessor, GroundingDinoForObjectDetection
from PIL import Image
import requests

# Increase the timeout for requests
requests.adapters.DEFAULT_RETRIES = 5
session = requests.Session()
session.mount('https://', requests.adapters.HTTPAdapter(max_retries=5))
session.get('https://huggingface.co', timeout=10)


url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
text = "a cat."

processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
model = GroundingDinoForObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny")

inputs = processor(images=image, text=text, return_tensors="pt")
outputs = model(**inputs)

# convert outputs (bounding boxes and class logits) to COCO API
target_sizes = torch.tensor([image.size[::-1]])
results = processor.image_processor.post_process_object_detection(
    outputs, threshold=0.35, target_sizes=target_sizes
)[0]
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 1) for i in box.tolist()]
    print(f"Detected {label.item()} with confidence " f"{round(score.item(), 2)} at location {box}")