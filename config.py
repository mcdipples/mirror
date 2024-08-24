import os

'''
curl -X POST -H "Content-Type: application/json" -d '{
  "image_url": "https://images.pexels.com/photos/106399/pexels-photo-106399.jpeg",
  "detector_prompt": "garage door",
  "inpainting_prompt": "Photorealistic image of a house exterior, focusing on where the garage door used to be. The garage door has been replaced with floor-to-ceiling glass doors, offering a clear view into a sleek, modern living room. The view is from outside looking in, during daytime with soft natural light. The living room features minimalist furniture, including a low-profile gray sofa, a glass coffee table, and abstract art on the walls. Warm wood flooring contrasts with white walls. Subtle reflections on the glass doors hint at the outdoor environment. The transition from exterior to interior is seamless, with the glass doors perfectly fitted into the existing house structure."
}' http://localhost:8080/process_image

$body = @{
    image_url = "https://images.pexels.com/photos/106399/pexels-photo-106399.jpeg"
    detector_prompt = "garage door"
    inpainting_prompt = "Photorealistic image of a house exterior, focusing on where the garage door used to be. The garage door has been replaced with floor-to-ceiling glass doors, offering a clear view into a sleek, modern living room. The view is from outside looking in, during daytime with soft natural light. The living room features minimalist furniture, including a low-profile gray sofa, a glass coffee table, and abstract art on the walls. Warm wood flooring contrasts with white walls. Subtle reflections on the glass doors hint at the outdoor environment. The transition from exterior to interior is seamless, with the glass doors perfectly fitted into the existing house structure."
} | ConvertTo-Json

Invoke-WebRequest -Uri "http://localhost:8080/process_image" -Method Post -Body $body -ContentType "application/json"
'''

# Constants
HOME = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(HOME, "downloader", "weights")
DATA_DIR = os.path.join(HOME, "data")
EDITED_IMAGES_DIR = os.path.join(DATA_DIR, "edits")

SAM_CHECKPOINT_PATH = os.path.join(WEIGHTS_DIR, "sam_vit_h_4b8939.pth")
SAM_MODEL_TYPE = "vit_h"
DINO_CHECKPOINT_PATH = os.path.join(WEIGHTS_DIR, "groundingdino_swinb_cogcoor.pth")

# DINO_CONFIG_PATH = os.path.join(HOME, "GroundingDINO", 
#                                 "groundingdino", "config", "GroundingDINO_SwinT_OGC.py")

DINO_CONFIG_PATH = os.path.join(HOME, "GroundingDINO", "groundingdino", "config", "GroundingDINO_SwinB_cfg.py")

# These are different model params for fine tuning ur predictions. This is for the GroundingDINO model.
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

# How many images you want dalle to generate
DALLE_IMAGE_COUNT = 3

"""
TODO:
- [ ] Make sure uploaded file is within dalle's image size
    - if not 1024x1024, resize it. if smaller than that, resize to the other dalle image sizes.
-[ ] IDEA: Create method to approve of the edited image and if so then basically start over with the new image.That way you can keep editing until you get to the point where you are happy with it.
- [ ] Prob wanna get a screen recording of this and then just show the edited image and then the original image next to it.
- [ ] Add conda stuff to README.md
"""