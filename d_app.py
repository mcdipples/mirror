import os
import torch
import cv2
import numpy as np
import supervision as sv
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from groundingdino.util.inference import Model, load_image, predict, annotate
from openai import OpenAI
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_manager import ModelManager, convert_to_bgr
from utils.mirror_utils import get_device, to_png, process_dalle_images, print_color

from config import *

def main():
    print_color("🔮 Oppy's MagicMirror 🔮", 'cyan')

    try:
        # Device selection
        use_gpu = input("Use GPU (if available)? (y/n): ").lower() == 'y'
        device = get_device() if use_gpu and torch.cuda.is_available() else torch.device("cpu")
        print_color(f"Using device: {device}", 'green')

        if input("Show Current Working Directory? (y/n): ").lower() == 'y':
            print_color(f"Current Working Directory: {os.getcwd()}", 'yellow')

        # Download weights
        if input("Download Weights? (y/n): ").lower() == 'y':
            try:
                ModelManager.download_SAM_weights()
                ModelManager.download_DINO_weights()
                print_color("Weights downloaded successfully!", 'green')
            except Exception as e:
                print_color(f"Error downloading weights: {e}", 'red')
                return

        # Image Loading
        image_path = input("Enter the path to your image: ")
        try:
            image = Image.open(image_path)
            print_color(f"Loaded image: {image_path}, Size: {image.size}", 'blue')
            image.show()
        except FileNotFoundError:
            print_color(f"Error: Image file not found at {image_path}", 'red')
            return
        except Exception as e:
            print_color(f"Error loading image: {e}", 'red')
            return

        # Image Editing Prompts
        print_color("🔧 Image Editing Prompts", 'purple')
        detector_prompt = input("What object do you want to edit? ")
        inplacing_prompt = input("What would you like to do with it? ")

        # Detection and Annotation
        try:
            dino_detector = ModelManager.load_DINO_model(device)
            mask_generator, sam_predictor = ModelManager.load_SAM_model(device)
        except Exception as e:
            print_color(f"Error loading models: {e}", 'red')
            return

        if detector_prompt:
            try:
                detections, labels = ModelManager.detect_objects(
                    detector_prompt=detector_prompt,
                    image=image,
                    dino=dino_detector
                )

                # Convert PIL Image to numpy array
                img_array = np.array(image)
                img_array = convert_to_bgr(img_array)

                # Convert detections to masks
                detections.mask = ModelManager.segment(
                    sam_predictor=sam_predictor,
                    image=cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB),
                    xyxy=detections.xyxy
                )

                box_annotator = sv.BoxAnnotator()
                mask_annotator = sv.MaskAnnotator()

                # Annotate image with detections
                print_color("Annotating image with detections...", 'green')
                annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
                annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
                print_color("Image annotated successfully!", 'green')

                # Convert numpy array to PIL Image and display
                annotated_pil = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
                annotated_pil.show()

                # Image Editing
                if inplacing_prompt:
                    print_color("🎭Creating mask...", 'green')
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
                    new_mask.save("temp_mask.png")

                    # OpenAI API Key
                    OPENAI_API_KEY = input("🔑Enter your OpenAI API Key: ")
                    client = OpenAI(api_key=OPENAI_API_KEY)

                    edit_image = to_png(image)

                    try:
                        print_color("Editing image...", 'purple')
                        edit_response = client.images.edit(
                            image=open(edit_image, "rb"),
                            mask=open("temp_mask.png", "rb"),
                            prompt=inplacing_prompt,
                            n=3,
                            size="1024x1024",
                            response_format="url",
                        )
                        print_color("Image edited successfully!", 'green')

                        edited_paths = process_dalle_images(edit_response, "edited_image", EDITED_IMAGES_DIR)

                        print_color("Choose an edited image:", 'yellow')
                        for i, edit_path in enumerate(edited_paths):
                            print(f"{i+1}. {edit_path}")
                            Image.open(edit_path).show()

                        choice = int(input("Enter the number of the image you want to select: "))
                        if 1 <= choice <= len(edited_paths):
                            selected_image = Image.open(edited_paths[choice-1])
                            selected_image.show()
                            print_color("Edit complete!", 'green')
                        else:
                            print_color("Invalid choice. No image selected.", 'yellow')

                    except Exception as e:
                        print_color(f"Error editing image: {e}", 'red')

                    # Clean up
                    if os.path.exists("temp_mask.png"):
                        os.remove("temp_mask.png")

            except Exception as e:
                print_color(f"Error during object detection or image processing: {e}", 'red')

        else:
            print_color("No detections available. Please run object detection first.", 'yellow')

    except KeyboardInterrupt:
        print_color("\nProgram interrupted by user.", 'yellow')
    except Exception as e:
        print_color(f"An unexpected error occurred: {e}", 'red')

if __name__ == "__main__":
    main()