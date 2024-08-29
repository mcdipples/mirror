import os
import sys
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.mirror_utils import print_color
from config import *

# Ensure the weights directory exists
os.makedirs(WEIGHTS_DIR, exist_ok=True)

def download_file(url, dest_path):
    session = requests.Session()
    retry = Retry(
        total=5,  # Number of retries
        backoff_factor=1,  # Wait 1, 2, 4, 8, 16 seconds between retries
        status_forcelist=[500, 502, 503, 504]  # Retry on these HTTP status codes
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    try:
        with session.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            with open(dest_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
    except requests.exceptions.RequestException as e:
        print_color(f"Error downloading file: {e}", "red")
        raise

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
    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    print_color("Downloading SAM weights...", "blue")
    try:
        download_file(url, SAM_CHECKPOINT_PATH)
        print_color("✅SAM weights downloaded successfully.", "green")
        print_color(f"SAM weights path: {SAM_CHECKPOINT_PATH}", "cyan")
    except Exception as e:
        print_color(f"Failed to download SAM weights: {e}", "red")
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
    url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth"
    print_color("Downloading DINO weights...", "blue")
    try:
        download_file(url, DINO_CHECKPOINT_PATH)
        print_color("✅DINO weights downloaded successfully.", "green")
        print_color(f"DINO weights path: {DINO_CHECKPOINT_PATH}", "cyan")
    except Exception as e:
        print_color(f"Failed to download DINO weights: {e}", "red")
# ------------------------------------------------------------

try:
    download_SAM_weights()
    download_DINO_weights()
except Exception as e:
    print_color(f"An error occurred during weight download: {e}", "red")
    sys.exit(1)