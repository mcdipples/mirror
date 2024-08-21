# import streamlit as st
# from PIL import Image
# import os
# import torch
# import cv2
# import numpy as np
# import supervision as sv

# from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
# from groundingdino.util.inference import Model, load_image, predict, annotate
# from openai import OpenAI

# from model_manager import ModelManager, convert_to_bgr
# from mirror_utils import get_device, to_png, process_dalle_images

# from config import *

# def main():
#     st.title("🔮 Oppy's MagicMirror 🔮")

# # ===== Main App & setup =====
#     # Device selection
#     if 'use_gpu' not in st.session_state:
#         st.session_state.use_gpu = False

#     # Use the session state for the checkbox
#     use_gpu = st.checkbox("Use GPU (if available)", value=st.session_state.use_gpu, key='use_gpu')
#     device = get_device() if use_gpu and torch.cuda.is_available() else torch.device("cpu")
#     st.write(f"Using device: {device}")

#     if st.button("Show Current Working Directory"):
#         st.write(f"Current Working Directory: {os.getcwd()}")

#     # Download weights
#     if st.button("Download Weights"):
#         ModelManager.download_SAM_weights()
#         ModelManager.download_DINO_weights()
#         st.success("Weights downloaded successfully!")

#     if 'uploaded_file' not in st.session_state:
#         st.session_state.uploaded_file = None

# # ==== Image Loading =====
#     uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
#     if uploaded_file is not None:
#         st.session_state.uploaded_file = uploaded_file

#     # Display and process image if it exists in session state
#     if st.session_state.uploaded_file is not None:
#         # if 'edited_image' in st.session_state:
#         #     image = st.session_state.edited_image
#         # else:
#         image = Image.open(st.session_state.uploaded_file)

#         col1, col2 = st.columns(2)
#         with col1:
#             if 'annotated_frame' in st.session_state:
#                 st.image(st.session_state.annotated_frame, caption="Detections 👆", use_column_width=True)
#             else:
#                 st.image(image, caption=f"Filename: {st.session_state.uploaded_file.name}, Size: {image.size}", use_column_width=True)
# # ===== END Image Loading =====

# # ===== Image Editing Prompts =====
#         with col2:
#             st.subheader("🔧 Image Editing Prompts")

#             if 'detector_prompt' not in st.session_state:
#                 st.session_state.detector_prompt = ""

#             detector_prompt = st.text_input("What object do you want to edit?", value=st.session_state.detector_prompt)
#             if detector_prompt != st.session_state.detector_prompt:
#                 st.session_state.detector_prompt = detector_prompt
#                 if 'annotated_frame' in st.session_state:
#                     del st.session_state.annotated_frame
#                 if 'detections' in st.session_state:
#                     del st.session_state.detections
#                 if 'labels' in st.session_state:
#                     del st.session_state.labels
#                 if os.path.exists("temp_mask.png"):
#                     os.remove("temp_mask.png")
#                 if 'edited_image' in st.session_state:
#                     del st.session_state.edited_image
#                 if 'edit_response' in st.session_state:
#                     del st.session_state.edit_response
#             if detector_prompt:
#                 st.write(f"Object to edit: {detector_prompt}")
            
#             if 'inplacing_prompt' not in st.session_state:
#                 st.session_state.inplacing_prompt = ""

#             inplacing_prompt = st.text_input("What would you like to do with it?", value=st.session_state.inplacing_prompt)
#             if inplacing_prompt != st.session_state.inplacing_prompt:
#                 st.session_state.inplacing_prompt = inplacing_prompt
#                 if 'edited_image' in st.session_state:
#                     print(f"\033[92m" + f"Deleting edited image..." + "\033[0m")
#                     del st.session_state.edited_image
#                 if 'edit_response' in st.session_state:
#                     del st.session_state.edit_response
#             if inplacing_prompt:
#                 st.write(f"Action to perform: {inplacing_prompt}")
# # ===== END Image Editing Prompts =====

# # ===== Detection and Annotation =====
#         # Load DINO model
#         dino_detector = ModelManager.load_DINO_model(device)

#         # Load SAM model
#         mask_generator, sam_predictor = ModelManager.load_SAM_model(device)

#         # Call detect_objects function if detector_prompt exists
#         if st.session_state.detector_prompt:
#             detections, labels = ModelManager.detect_objects(
#                 detector_prompt=st.session_state.detector_prompt,
#                 image=image,
#                 dino=dino_detector
#             )
#             st.session_state.detections = detections
#             st.session_state.labels = labels

#         if 'detections' in st.session_state and 'labels' in st.session_state:
#             # Convert PIL Image to numpy array
#             img_array = np.array(image)

#             # Convert to BGR format
#             img_array = convert_to_bgr(img_array)

#             # Convert detections to masks
#             st.session_state.detections.mask = ModelManager.segment(
#                 sam_predictor=sam_predictor,
#                 image=cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB),
#                 xyxy=st.session_state.detections.xyxy
#             )

#             # Create annotators if not already in session state
#             if 'box_annotator' not in st.session_state:
#                 st.session_state.box_annotator = sv.BoxAnnotator()
#             if 'mask_annotator' not in st.session_state:
#                 st.session_state.mask_annotator = sv.MaskAnnotator()
            
#             box_annotator = st.session_state.box_annotator
#             mask_annotator = st.session_state.mask_annotator


#             # Save the annotated frame to session state
#             if 'annotated_frame' not in st.session_state:
#                 # Annotate image with detections
#                 print(f"\033[92m" + f"Annotating image with detections: {st.session_state.detections}" + "\033[0m")
#                 annotated_image = mask_annotator.annotate(scene=image.copy(), detections=st.session_state.detections)
#                 annotated_image = box_annotator.annotate(scene=annotated_image, detections=st.session_state.detections)
#                 st.session_state.annotated_frame = annotated_image
#                 print("\033[92m" + "Image annotated successfully!" + "\033[0m")
#                 st.rerun()
#             else:
#                 print("\033[95m" + "No new annotation." + "\033[0m")
# # ===== END Detection and Annotation =====

# # ========= Image Editing =========
#             # Process the mask for editing
#             if 'inplacing_prompt' in st.session_state and st.session_state.inplacing_prompt != "":
#                 if not os.path.exists("temp_mask.png"):
#                     print(f"\033[92m" + f"🎭Creating mask..." + "\033[0m")
#                     chosen_mask = st.session_state.detections.mask[0]
#                     chosen_mask = chosen_mask.astype("uint8")
#                     chosen_mask[chosen_mask != 0] = 255
#                     chosen_mask[chosen_mask == 0] = 1
#                     chosen_mask[chosen_mask == 255] = 0
#                     chosen_mask[chosen_mask == 1] = 255

#                     width, height = image.size
#                     mask = Image.new("RGBA", (width, height), (0, 0, 0, 1))
#                     pix = np.array(mask)
#                     pix[:, :, 3] = chosen_mask
#                     new_mask = Image.fromarray(pix, "RGBA")

#                     # Save the mask temporarily
#                     new_mask.save("temp_mask.png")

#                 # Check if OpenAI API key exists and is not empty in session state
#                 if 'openai_api_key' not in st.session_state or st.session_state.openai_api_key == "":
#                     st.warning("Please enter your OpenAI API key to proceed with image editing.")
#                     st.session_state.openai_api_key = st.text_input("🔑Enter your OpenAI API Key 🔑", type="password")
                
#                 # If API key exists and is not empty, proceed with editing operations
#                 if st.session_state.openai_api_key:
#                     OPENAI_API_KEY = st.session_state.openai_api_key

#                     print(f"\033[92m" + f"API Key: {OPENAI_API_KEY}" + "\033[0m")
#                     print(f"\033[92m" + f"Session state key: {st.session_state.openai_api_key}" + "\033[0m")

#                     if 'openai_client' not in st.session_state:
#                         st.session_state.openai_client = OpenAI(api_key=OPENAI_API_KEY)
#                     client = st.session_state.openai_client

#                     edit_image = to_png(image)
                    
#                     edit_response = None

#                     # Print all current session_state variables to the console
#                     print("\033[94m" + "Current session_state variables:" + "\033[0m")
#                     for key, value in st.session_state.items():
#                         print(f"\033[94m{key}: {value}\033[0m")
#                     print("\033[94m" + "End of session_state variables" + "\033[0m")

#                     # Check if edit_response exists in session state
#                     if 'edit_response' not in st.session_state and 'edited_image' not in st.session_state:
#                         # Edit the image
#                         try:    
#                             print(f"\033[95m" + f"Editing image..." + "\033[0m")
#                             edit_response = client.images.edit(
#                                 image=open(edit_image, "rb"),
#                                 mask=open("temp_mask.png", "rb"),
#                                 prompt=st.session_state.inplacing_prompt,
#                                 n=3,
#                                 size="1024x1024",
#                                 response_format="url",
#                             )

#                             st.session_state.edit_response = edit_response
#                             print(f"\033[95m" + f"Got it~" + "\033[0m")
#                         except Exception as e:
#                             print(f"\033[91m" + f"Error editing image: {e}" + "\033[0m")
#                             st.error(f"Error editing image: {e}")
                    
#                     # ===== END Image Editing =====

#                     # ===== Select Edited Image =====
#                     if 'edit_response' in st.session_state and 'edited_image' not in st.session_state:

#                         edited_paths = process_dalle_images(st.session_state.edit_response, "edited_image", EDITED_IMAGES_DIR)

#                         # Display the edited images
#                         st.subheader("Choose an edited image:")
#                         cols = st.columns(len(edited_paths))

#                         print(f"\033[92m" + f"Edited paths: {edited_paths}" + "\033[0m")
#                         for i, edit_path in enumerate(edited_paths):
#                             with cols[i]:
#                                 try:
#                                     im = Image.open(edit_path)
#                                     st.image(im, use_column_width=True)
#                                 except Exception as e:
#                                     st.error(f"Error opening image: {e}")
#                                 if st.button(f"Select Image {i+1}"):
#                                     print(f"\033[92m" + f"Selecting image {i+1}..." + "\033[0m")
#                                     st.session_state.edited_image = im
#                                     if 'annotated_frame' in st.session_state:
#                                         del st.session_state.annotated_frame
#                                     if os.path.exists("temp_mask.png"):
#                                         os.remove("temp_mask.png")
#                                     if 'edit_response' in st.session_state:
#                                         del st.session_state.edit_response
#                     # ===== END Select Edited Image =====

#                         if st.button("Regenerate"):
#                             if 'edit_response' in st.session_state:
#                                 del st.session_state.edit_response
#                             if 'edited_image' in st.session_state:
#                                 del st.session_state.edited_image
#                         # st.rerun()

#                     # Add undo option
#                     if 'edited_image' in st.session_state:
#                         st.image(st.session_state.edited_image, caption="Edited Image", use_column_width=True)
#                         if st.button("Undo Edit"):
#                             del st.session_state.edited_image
#                             if os.path.exists("temp_mask.png"):
#                                 os.remove("temp_mask.png")
#                             if 'edit_response' in st.session_state:
#                                 del st.session_state.edit_response
#                             # st.rerun()

#         else:
#             st.warning("No detections available. Please run object detection first.")


# if __name__ == "__main__":
#     main()