import os
import requests
import torch
import tempfile
import numpy as np
import cv2

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from groundingdino.util.inference import Model
from utils.mirror_utils import print_color

from config import *

# ------------------------------------------------------------
# FUNCTION: convert_to_bgr
# This function converts the image array to BGR format if it's in RGB format.
# PARAMS:
#     - img_array: The image array to convert.
# RETURNS:
#     - The converted image array (BGR format).
# ------------------------------------------------------------
def convert_to_bgr(img_array):
    # If the image is in RGB format (PIL default), convert it to BGR (which is what OpenCV uses)
    if img_array.shape[2] == 3:  # If it's a 3-channel image
        return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    elif img_array.shape[2] == 4:  # If it has an alpha channel
        return cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
    return img_array  # Return original array if it doesn't match these conditions

class ModelManager:

    # ------------------------------------------------------------
    # '''
    # FUNCTION: get_temp_weights_dir
    # This function returns the path to the temporary weights directory.
    # '''
    # ------------------------------------------------------------
    @staticmethod
    def get_temp_weights_dir():
        return tempfile.gettempdir()

    # ------------------------------------------------------------
    # '''
    # FUNCTION: download_weights
    # This function downloads the SAM weights from the FB AI public files.
    # '''
    # ------------------------------------------------------------
    @staticmethod
    def download_SAM_weights():
        if os.path.isfile(SAM_CHECKPOINT_PATH):
            print_color("✅SAM weights already downloaded.", "green")
            return
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        print_color("Downloading SAM weights...", "blue")
        response = requests.get(url)
        with open(SAM_CHECKPOINT_PATH, 'wb') as f:
            f.write(response.content)
        print_color("✅SAM weights downloaded successfully.", "green")
        print_color(f"SAM weights path: {SAM_CHECKPOINT_PATH}", "cyan")
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # '''
    # FUNCTION: download_DINO_weights
    # This function downloads the DINO weights from the GitHub repository.
    # '''
    # ------------------------------------------------------------
    @staticmethod
    def download_DINO_weights():
        if os.path.isfile(DINO_CHECKPOINT_PATH):
            print_color("✅DINO weights already downloaded.", "green")
            return
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
        print_color("Downloading DINO weights...", "blue")
        response = requests.get(url)
        with open(DINO_CHECKPOINT_PATH, 'wb') as f:
            f.write(response.content)
        print_color("✅DINO weights downloaded successfully.", "green")
        print_color(f"DINO weights path: {DINO_CHECKPOINT_PATH}", "cyan")
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # '''
    # FUNCTION: load_SAM_model
    # This function loads the SAM model from the FB AI public files.
    # PARAMS:
    #     - device: The device to load the model on (CPU or GPU).
    # RETURNS:
    #     - mask_generator: The SAM mask generator.
    #     - sam_predictor: The SAM predictor.
    # '''
    # ------------------------------------------------------------
    @staticmethod
    def load_SAM_model(device):
        sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT_PATH)
        sam.to(device=device)

        mask_generator = SamAutomaticMaskGenerator(sam)
        sam_predictor = SamPredictor(sam)
        print_color("✅SAM model loaded successfully!", "green")
        return mask_generator, sam_predictor
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # '''
    # FUNCTION: load_DINO_model
    # This function loads the DINO model from the GitHub repository.
    # PARAMS:
    #     - device: The device to load the model on (CPU or GPU).
    # RETURNS:
    #     - dino: The DINO model.
    # '''
    # ------------------------------------------------------------
    @staticmethod
    def load_DINO_model(device):
        dino = Model(model_config_path=DINO_CONFIG_PATH, model_checkpoint_path=DINO_CHECKPOINT_PATH)
        print_color("✅DINO model loaded successfully!", "green")
        return dino
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # '''
    # FUNCTION: detect_objects
    # This function detects objects in the image using the DINO model.
    # PARAMS:
    #     - detector_prompt: The prompt to detect objects with.
    #     - image: The image to detect objects in.
    #     - dino: The DINO model.
    #     - box_threshold: The threshold for bounding box detection.
    #     - text_threshold: The threshold for text detection.
    # RETURNS:
    #     - detections: The detections in the image.
    #     - labels: The labels of the detections.
    # '''
    # ------------------------------------------------------------
    @staticmethod
    def detect_objects(detector_prompt, image, dino, box_threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD):
        print_color("Detecting objects...", "blue")
        
        # Convert PIL Image to numpy array
        img_array = np.array(image)
        
        img_array = convert_to_bgr(img_array)
        
        # Perform object detection
        detections, labels = dino.predict_with_caption(
            image=img_array,
            caption=detector_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )

        if len(detections) == 0 or len(labels) == 0:
            print_color("😰 No objects detected.", "yellow")
        else:
            print_color("✅Object detection completed!", "green")

        detections.class_id = np.arange(len(detections), dtype=int)
        
        return detections, labels
    # ------------------------------------------------------------
    # '''
    # FUNCTION: segment
    # This function segments the image using the SAM model.
    # PARAMS:
    #     - sam_predictor: The SAM predictor.
    #     - image: The image to segment (RGB, numpy array).
    #     - xyxy: The bounding boxes to segment (xyxy format, numpy array).
    # RETURNS:
    #     - The segmented image.
    # '''
    # ------------------------------------------------------------
    @staticmethod
    def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        sam_predictor.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = sam_predictor.predict(
                box=box,
                multimask_output=True
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)
    # ------------------------------------------------------------
