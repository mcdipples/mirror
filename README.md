This application uses Streamlit as a basic Gui framework to provide a streamlined user experience when uploading images and prompts for automated object detection and replacement within the uploaded image.

Please make sure following the installation steps strictly, otherwise the program may produce: 
```bash
NameError: name '_C' is not defined
```

If this happens, please reinstall the groundingDINO by recloning the git and do all the installation steps again.

#### how to check cuda:
```bash
echo $CUDA_HOME
```
If it prints nothing, then it means you haven't set up the path.

Run this so the environment variable will be set under current shell. 
```bash
export CUDA_HOME=/path/to/cuda-11.3
```

Notice the version of cuda should be aligned with your CUDA runtime, for there might exists multiple cuda at the same time. 

If you want to set the CUDA_HOME permanently, store it using:

```bash
echo 'export CUDA_HOME=/path/to/cuda' >> ~/.bashrc
```
after that, source the bashrc file and check CUDA_HOME:
```bash
source ~/.bashrc
echo $CUDA_HOME
```

In this example, /path/to/cuda-11.3 should be replaced with the path where your CUDA toolkit is installed. You can find this by typing **which nvcc** in your terminal:

For instance, 
if the output is /usr/local/cuda/bin/nvcc, then:
```bash
export CUDA_HOME=/usr/local/cuda
```

**Installation:**
1.Clone the GroundingDINO repository from GitHub.
```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git
```

2. Change the current directory to the GroundingDINO folder.

```bash
cd GroundingDINO/
pip install -e .
```

