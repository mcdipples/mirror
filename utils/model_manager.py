import os
import sys
import requests
import torch
import tempfile
import numpy as np
import cv2

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from GroundingDINO.groundingdino.util.inference import Model
from GroundingDINO.groundingdino.util import box_ops

from utils.mirror_utils import print_color

from config import *

def convert_to_bgr(img_array):
    """
    Convert the image array to BGR format if it's in RGB format.

    Args:
        img_array (np.ndarray): The image array to convert.

    Returns:
        np.ndarray: The converted image array (BGR format).
    """
    if img_array.shape[2] == 3:  # If it's a 3-channel image
        return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    elif img_array.shape[2] == 4:  # If it has an alpha channel
        return cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
    return img_array  # Return original array if it doesn't match these conditions

class ModelManager:
    @staticmethod
    def load_SAM_model(device):
        """
        Load the SAM model from the FB AI public files.

        Args:
            device (str): The device to load the model on (CPU or GPU).

        Returns:
            tuple: The SAM mask generator and the SAM predictor.
        """
        try:
            sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT_PATH)
            sam.to(device=device)

            mask_generator = SamAutomaticMaskGenerator(sam)
            sam_predictor = SamPredictor(sam)
            print_color("✅SAM model loaded successfully!", "green")
            return mask_generator, sam_predictor
        except Exception as e:
            print_color(f"❌ Error loading SAM model: {str(e)}", "red")
            return None, None

    @staticmethod
    def load_DINO_model(device):
        """
        Load the DINO model from the GitHub repository.

        Args:
            device (str): The device to load the model on (CPU or GPU).

        Returns:
            Model: The DINO model.
        """
        try:
            dino = Model(model_config_path=DINO_CONFIG_PATH, model_checkpoint_path=DINO_CHECKPOINT_PATH)
            print_color("✅DINO model loaded successfully!", "green")
            return dino
        except Exception as e:
            print_color(f"❌ Error loading DINO model: {str(e)}", "red")
            return None
        
    def detect_objects(self, detector_prompt, image, box_threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD):
        """
        Detect objects in the image using the DINO model.

        Args:
            detector_prompt (str): The prompt to detect objects with.
            image (PIL.Image.Image): The image to detect objects in.
            box_threshold (float): The threshold for bounding box detection.
            text_threshold (float): The threshold for text detection.

        Returns:
            tuple: The detections and the labels of the detections.
        """
        try:
            print_color("Detecting objects...", "blue")
            
            # Convert PIL Image to numpy array
            img_array = np.array(image)
            
            img_array = convert_to_bgr(img_array)
            
            # Perform object detection
            detections, labels = self.dino.predict_with_caption(
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

        except Exception as e:
            print_color(f"❌ Error during object detection: {str(e)}", "red")
            return None, None
        return detections, labels

    def segment(self, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        """
        Segment the image using the SAM model. Returns the masks of the detected objects.

        Args:
            image (np.ndarray): The image to segment (BGR, numpy array).
            xyxy (np.ndarray): The bounding boxes to segment (xyxy format, numpy array).

        Returns:
            np.array: The highest confidence mask of the detected objects.
        """

        """
        def set_image(
        self,
        image: np.ndarray,
        image_format: str = "RGB",
    ) -> None:
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method.

        Arguments:
          image (np.ndarray): The image for calculating masks. Expects an image in HWC uint8 format, with pixel values in [0, 255].
          image_format (str): The color format of the image, in ['RGB', 'BGR'].
        """

        
        self.sam_predictor.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = self.sam_predictor.predict(
                box=box,
                multimask_output=True
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)

# ~~~~~~~~~~~~~~~~ 🧶🧶 ~~~~~~~~~~~~~~~~~~~

    def detect_and_segment(self, image: np.ndarray, detector_prompt: str, save_path: str) -> np.ndarray:
        """
        Detect and segment the image using the SAM model.
        """
        detections, labels = self.detect_objects(detector_prompt, image)
        self.editor["detections"] = detections
        self.editor["labels"] = labels  

        assert !isinstance(image, np.ndarray), "Expects an image in HWC uint8 format, with pixel values in [0, 255]. Please pass in an appropriate numpy array."

        # Convert PIL Image to numpy array and then to BGR
        image_source = Image.open(image_path).convert("RGB")
    image = np.asarray(image_source)


        return self.segment(image, xyxy)

###
#TODO: Store segmentation results (masks) as self.masks and self.detections but then clear them after processing the image in flask_api.py
# also make sure to call SamPredictor.reset_image() after processing the image in flask_api.py
###
    def annotate(self, image: np.ndarray, mask_path: str) -> np.ndarray:
        """
        Annotate the image with the detected objects.

        Args:
            image (np.ndarray): The image to annotate (RGB, numpy array).
            mask_path (str): The path to the mask file.

        Returns:
            np.ndarray: The annotated image.
        """
        # Load the mask from the file
        mask = Image.open(mask_path)
        mask = np.array(mask)

        # Create annotators if not already created
        if self.box_annotator is None:
            self.box_annotator = sv.BoxAnnotator()
        if self.mask_annotator is None:
            self.mask_annotator = sv.MaskAnnotator()

        # Annotate image with detections
        print(f"\033[92m" + f"Annotating image with detections (mask values): {mask}" + "\033[0m")
        annotated_image = self.mask_annotator.annotate(scene=image.copy(), detections=mask)
        annotated_image = self.box_annotator.annotate(scene=annotated_image, detections=mask)
        print("\033[92m" + "Image annotated successfully!" + "\033[0m")

        return annotated_image
    
    def __init__(self, dino_checkpoint: str, dino_config: str, sam_checkpoint: str) -> None:
        """
        Initialize the ModelManager class.

        Args:
            dino_checkpoint (str): The path to the DINO checkpoint.
            dino_config (str): The path to the DINO configuration.
            sam_checkpoint (str): The path to the SAM checkpoint.
        """
        self.dino_checkpoint = dino_checkpoint
        self.dino_config = dino_config
        self.sam_checkpoint = sam_checkpoint

        self.editor = {
            "masks": [],
            "detections": [],
            "labels": []
        }

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.dino = self.load_DINO_model(self.device)
        self.sam_predictor = self.load_SAM_model(self.device)
        self.box_annotator = None
        self.mask_annotator = None
