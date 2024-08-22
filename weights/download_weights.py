import os
import requests

from utils.mirror_utils import print_color
from config import *

# ------------------------------------------------------------
# '''
# FUNCTION: download_weights
# This function downloads the SAM weights from the FB AI public files.
# '''
# ------------------------------------------------------------
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
def download_DINO_weights():
    if os.path.isfile(DINO_CHECKPOINT_PATH):
        print_color("✅DINO weights already downloaded.", "green")
        return
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth"
    print_color("Downloading DINO weights...", "blue")
    response = requests.get(url)
    with open(DINO_CHECKPOINT_PATH, 'wb') as f:
        f.write(response.content)
    print_color("✅DINO weights downloaded successfully.", "green")
    print_color(f"DINO weights path: {DINO_CHECKPOINT_PATH}", "cyan")
# ------------------------------------------------------------

download_SAM_weights()
download_DINO_weights()