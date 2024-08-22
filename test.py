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


import argparse
import os
import copy

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_convert

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict

import supervision as sv

# segment anything
from segment_anything import build_sam, SamPredictor 
import cv2
import numpy as np
import matplotlib.pyplot as plt


# diffusers
import PIL
import requests
import torch
from io import BytesIO
from diffusers import StableDiffusionInpaintPipeline


from huggingface_hub import hf_hub_download

from transformers import AutoProcessor, GroundingDinoForObjectDetection
from PIL import Image
import requests

sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))


def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file) 
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model   

# Use this command for evaluate the Grounding DINO model
# Or you can download the model by yourself
ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"

groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename)


# # Increase the timeout for requests
# requests.adapters.DEFAULT_RETRIES = 5
# session = requests.Session()
# session.mount('https://', requests.adapters.HTTPAdapter(max_retries=5))
# session.get('https://huggingface.co', timeout=10)


# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
# text = "a cat."

# processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
# model = GroundingDinoForObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny")

# inputs = processor(images=image, text=text, return_tensors="pt")
# outputs = model(**inputs)

# # convert outputs (bounding boxes and class logits) to COCO API
# target_sizes = torch.tensor([image.size[::-1]])
# results = processor.image_processor.post_process_object_detection(
#     outputs, threshold=0.35, target_sizes=target_sizes
# )[0]
# for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
#     box = [round(i, 1) for i in box.tolist()]
#     print(f"Detected {label.item()} with confidence " f"{round(score.item(), 2)} at location {box}")