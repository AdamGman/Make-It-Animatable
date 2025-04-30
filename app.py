import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.system("bash install.sh")

# Import app from the current directory
from app import *

# Initialize models and Gradio interface
init_models()
demo = init_blocks()

# Launch Gradio
demo.launch(
    server_name="0.0.0.0", 
    server_port=7860, 
    allowed_paths=["."], 
    show_error=True
)
