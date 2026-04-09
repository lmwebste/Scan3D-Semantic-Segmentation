# Scan3D-Semantic-Segmentation

# First-Time Setup
To compile the C++ wrappers for your host system, open a terminal in this directory and run the following code:
sh "cpp_wrappers/compile_wrappers.sh"

The trained model can be downloaded from https://drive.google.com/file/d/17F43XiFFfF2gQ_jFnJnQ59V47LZAfORq/view?usp=drive_link; it should be placed in "train/checkpoints/", alongside the placeholder README file therein.

Ensure the dependencies listed in requirements.txt are available, and that the system supports CUDA (i.e., your computer has an NVIDIA GPU; this has been tested for CUDA 12.6)
