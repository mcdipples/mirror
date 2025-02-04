import os
import sys
import traceback
import requests
import torch
import tempfile
import numpy as np
import cv2
from PIL import Image

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

# Add the parent directory of GroundingDINO to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'GroundingDINO')))

from groundingdino.util.inference import Model
from groundingdino.util import box_ops

from utils.mirror_utils import print_color
from utils.image_processing import to_png
from config import *

# -------------------------------------------------
class ModelManager:
    def __init__(self, dino_checkpoint: str, dino_config: str, sam_checkpoint: str) -> None:
        """
        Initialize the ModelManager class.

        Make sure weights are downloaded before initializing the ModelManager.

        Args:
            dino_checkpoint (str): The path to the DINO checkpoint.
            dino_config (str): The path to the DINO configuration.
            sam_checkpoint (str): The path to the SAM checkpoint.
        """
        print_color("Initializing ModelManager...", "blue")
        try:
            self.dino_checkpoint = dino_checkpoint
            self.dino_config = dino_config
            self.sam_checkpoint = sam_checkpoint

            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            self.dino = self.load_DINO_model(self.device)
            _, self.sam_predictor = self.load_SAM_model(self.device)
            self.box_annotator = None
            self.mask_annotator = None
            self.clear_detections()
            print_color("ModelManager initialized successfully!", "green")
        except Exception as e:
            print_color(f"❌ Error initializing ModelManager: {str(e)}", "red")

    @staticmethod
    def load_SAM_model(device) -> tuple[SamAutomaticMaskGenerator, SamPredictor]:
        """
        Load the SAM model from the FB AI public files.

        Args:
            device (str): The device to load the model on (CPU or GPU).

        Returns:
            tuple: The SAM mask generator and the SAM predictor.
        """
        print_color("Loading SAM model...", "blue")
        try:
            sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT_PATH)
            sam.to(device=device)

            mask_generator = SamAutomaticMaskGenerator(sam)
            sam_predictor = SamPredictor(sam)
            print_color("✅ SAM model loaded successfully!", "green")
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
        print_color("Loading DINO model...", "blue")
        try:
            dino = Model(model_config_path=DINO_CONFIG_PATH, model_checkpoint_path=DINO_CHECKPOINT_PATH, device=device)
            print_color("✅ DINO model loaded successfully!", "green")
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
        print_color("Detecting objects...", "blue")
        try:
            # Convert PIL Image to numpy array
            img_array = np.array(image)
            img_array_bgr = img_array.astype(np.uint8)
            
            # Perform object detection
            detections, labels = self.dino.predict_with_caption(
                image=img_array_bgr,
                caption=detector_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold
            )

            if len(detections) == 0 or len(labels) == 0:
                print_color("😰 No objects detected.", "yellow")
            else:
                print_color("✅ Object detection completed!", "green")

            detections.class_id = np.arange(len(detections), dtype=int)
            print_color(f"Detections: {detections}", "gray")
            print_color(f"Labels: {labels}", "gray")
            
            return detections, labels
        except Exception as e:
            print_color(f"❌ Error during object detection: {str(e)}", "red")
            return None, None

    def segment(self, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        """
        Segment the image using the SAM model. Returns the masks of the detected objects.

        Args:
            image (np.ndarray): The image for calculating masks (HWC uint8 format, range [0, 255]).
            xyxy (np.ndarray): The bounding boxes to segment (xyxy format, numpy array).

        Returns:
            np.array: The highest confidence mask of the set of detected objects.
        """
        print_color("Segmenting image...", "blue")
        try:
            if not isinstance(image, np.ndarray):
                raise ValueError("ModelManager.segment expects an image in HWC uint8 format, with pixel values in [0, 255]. Please pass in an appropriate numpy array.")

            print_color("Setting image for SAM predictor...", "blue")
            self.sam_predictor.set_image(image)
            result_masks = []

            print_color(f"Number of bounding boxes: {len(xyxy)}", "blue")
            print_color("Predicting masks with SAM...", "blue")
            for box in xyxy:
                # predict masks with sam (outputs 3 masks for each box, 2nd and 3rd dims are the image size)
                masks, scores, logits = self.sam_predictor.predict(
                    box=box,
                    multimask_output=True
                )

                print_color(f"Masks shape: {masks.shape}, Scores: {scores}", "gray")

                print_color("Selecting highest confidence mask...", "blue")
                index = np.argmax(scores)
                # TODO: allow for multiple masks
                # result_masks.append(masks[index])
                result_mask = masks[index]

            print_color("✅ Segmentation completed!", "green")
            print_color(f"Result masks shape: {result_mask.shape}", "gray")
            return result_mask
        except Exception as e:
            print_color(f"❌ Error during segmentation: {str(e)}", "red")
            return np.array([])

    def detect_and_segment(self, image: Image.Image, detector_prompt: str) -> np.ndarray:
        """
        Detect and segment the image using the SAM model. Saves the masks to the save_path specified.
        
        Args:
            image (Image.Image): The image to process.
            detector_prompt (str): The prompt for the object detector.
        
        Returns:
            np.ndarray: The segmented masks (binary masks).

        ♻️ TODO: allow for multiple masks
        """
        print_color("Detecting and segmenting image...", "blue")
        try:
            # Detect objects in the image (takes in PIL Image and returns detections and labels)
            detections, labels = self.detect_objects(detector_prompt, image)

            if detections is None or labels is None:
                raise ValueError("No objects detected in the image.")

            print_color(f"Detections: {detections}", "gray")
            print_color(f"Labels: {labels}", "gray")

            self.editor["detections"] = detections
            self.editor["labels"] = labels  

            # Convert PIL Image to numpy array and then to RGB
            image = np.asarray(image.convert("RGB"))

            # Segment the image using the detected bounding boxes
            masks = self.segment(image, detections.xyxy)

            if masks.size == 0:
                raise ValueError("No masks generated during segmentation.")

            '''
            TODO: allow for multiple masks
            '''
            self.editor["masks"] = masks
            print_color("✅ Detect and segment completed!", "green")
            return masks
        except Exception as e:
            print_color(f"❌ Error during detect and segment: {str(e)}", "red")
            print_color(f"Error type: {type(e)}", "red")
            print_color(f"Error traceback: {traceback.format_exc()}", "red")
            return np.array([])

    def annotate(self, image: Image.Image, mask_path: str) -> Image.Image:
        """
        Annotate the image with the detected objects.

        Args:
            image (Image.Image): The image to annotate (RGB, numpy array).
            mask_path (str): The path to the mask file.

        Returns:
            Image.Image: The annotated image.

            # annotation example: give the mask to the model to annotate
             annotated_image = models.annotate(image, binary_mask)
             ImageShow.show(annotated_image)
        """
        print_color("Annotating image...", "blue")
        try:
            # Load the mask from the file
            mask = Image.open(mask_path)
            mask = np.array(mask)

            # Create annotators if not already created
            if self.box_annotator is None:
                self.box_annotator = sv.BoxAnnotator()
            if self.mask_annotator is None:
                self.mask_annotator = sv.MaskAnnotator()

            # Annotate image with detections
            print_color(f"Annotating image with detections (mask values): {mask}", "gray")

            annotated_image_1 = self.mask_annotator.annotate(scene=image.copy(), detections=self.editor["detections"])
            annotated_image_2 = self.box_annotator.annotate(scene=annotated_image_1, detections=self.editor["detections"])

            print_color("✅ Image annotated successfully!", "green")
            return annotated_image_2
        except Exception as e:
            print_color(f"❌ Error during annotation: {str(e)}", "red")
            return image

    def clear_detections(self) -> None:
        """
        Clear the editor by resetting the detections.
        """
        print_color("Clearing detections...", "blue")
        try:
            self.editor = {
                "masks": [],
                "detections": [],
                "labels": []
            }
            print_color("✅ Detections cleared!", "green")
        except Exception as e:
            print_color(f"❌ Error clearing detections: {str(e)}", "red")
# -------------------------------------------------