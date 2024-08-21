import os

# Constants
HOME = os.getcwd()
WEIGHTS_DIR = os.path.join(HOME, "weights")
DATA_DIR = os.path.join(HOME, "data")
EDITED_IMAGES_DIR = os.path.join(DATA_DIR, "edits")

SAM_CHECKPOINT_PATH = os.path.join(WEIGHTS_DIR, "sam_vit_h_4b8939.pth")
SAM_MODEL_TYPE = "vit_h"
DINO_CHECKPOINT_PATH = os.path.join(WEIGHTS_DIR, "groundingdino_swint_ogc.pth")

# ❗❗ THIS IS HARDCODED BUT YOU SHOULD BE ABLE TO JUST CHANGE YOUR ENVIRONMENT NAME 
# (assuming conda env) ❗❗
DINO_CONFIG_PATH = os.path.join(HOME, "GroundingDINO", 
                                "groundingdino", "config", "GroundingDINO_SwinT_OGC.py")

# These are different model params for fine tuning ur predictions.
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