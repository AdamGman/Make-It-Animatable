import os
import tempfile
import threading
import requests
import re
import subprocess
from fastapi import FastAPI, Request
import uvicorn

# This holds the API app
api_app = FastAPI()

def start_api_thread(pipeline_function, DB_class):
    """Start the API server in a background thread
    
    Args:
        pipeline_function: The _pipeline function from the main app
        DB_class: The DB class from the main app
    """
    api_thread = threading.Thread(
        target=run_api,
        args=(pipeline_function, DB_class),
        daemon=True
    )
    api_thread.start()
    return api_thread

def run_api(pipeline_function, DB_class):
    """Run the FastAPI server
    
    Args:
        pipeline_function: The _pipeline function from the main app
        DB_class: The DB class from the main app
    """
    global _pipeline_function, _DB_class
    _pipeline_function = pipeline_function
    _DB_class = DB_class
    uvicorn.run(api_app, host="0.0.0.0", port=8000)

def get_animation_path():
    """Get the path to the standard animation file"""
    # Use Standard Run.fbx as the default animation
    animation_path = os.path.join("data", "Standard Run.fbx")
    if not os.path.exists(animation_path):
        animation_path = os.path.join("data", "Mixamo", "animation", "Standard Run.fbx")
    
    if not os.path.exists(animation_path):
        raise FileNotFoundError("Standard Run.fbx animation file not found")
    
    return animation_path

def extract_filename_from_url(url):
    """Extract the filename from a URL
    
    Args:
        url: URL to extract filename from
        
    Returns:
        Filename extracted from URL
    """
    return url.split('/')[-1]

def upload_to_render_server(file_path, original_url=None):
    """Upload a file to the render.com server
    
    Args:
        file_path: Path to the file to upload
        original_url: Original URL of the model (used for naming)
        
    Returns:
        URL to the uploaded file if successful, None otherwise
    """
    try:
        # Determine filename to preserve name relationship
        if original_url:
            orig_filename = extract_filename_from_url(original_url)
            base_name = os.path.splitext(orig_filename)[0]
            ext = os.path.splitext(file_path)[1]
            new_filename = f"{base_name}_rigged{ext}"
        else:
            new_filename = os.path.basename(file_path)
        
        # Server endpoint
        upload_url = 'https://viverse-backend.onrender.com/api/upload-rigged-model'
        
        print(f"Uploading rigged model {file_path} to render.com server...")
        
        # Create multipart form-data payload
        with open(file_path, 'rb') as f:
            files = {'modelFile': (new_filename, f, 'application/octet-stream')}
            response = requests.post(upload_url, files=files)
        
        if response.status_code != 200:
            print(f"Error uploading file: {response.text}")
            return None
            
        result = response.json()
        
        if not result.get('success'):
            print(f"Upload failed: {result.get('error', 'Unknown error')}")
            return None
            
        return result.get('persistentUrl')
    except Exception as e:
        print(f"Error uploading file: {e}")
        return None

def convert_fbx_to_glb(fbx_path):
    """Convert FBX to GLB using FBX2glTF
    
    Args:
        fbx_path: Path to FBX file
        
    Returns:
        Path to GLB file if successful, None otherwise
    """
    if not fbx_path or not os.path.isfile(fbx_path) or not fbx_path.endswith('.fbx'):
        print(f"Invalid FBX path: {fbx_path}")
        return None
    
    # Create output GLB path from FBX path
    glb_path = fbx_path.replace('.fbx', '.glb')
    
    # Check different locations for FBX2glTF binary
    fbx2glTF_paths = [
        "./FBX2glTF",
        "./FBX2glTF.exe",
        "/usr/local/bin/FBX2glTF",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "FBX2glTF"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "FBX2glTF.exe"),
    ]
    
    fbx2glTF_path = None
    for path in fbx2glTF_paths:
        if os.path.isfile(path):
            fbx2glTF_path = path
            break
    
    if not fbx2glTF_path:
        print("FBX2glTF binary not found")
        return None
    
    # Run FBX2glTF
    try:
        cmd = [fbx2glTF_path, "--input", fbx_path, "--output", os.path.dirname(glb_path)]
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error converting FBX to GLB: {result.stderr}")
            return None
        
        print(f"Successfully converted FBX to GLB: {glb_path}")
        return glb_path
    except Exception as e:
        print(f"Error converting FBX to GLB: {e}")
        return None

def download_file(url):
    """Downloads a file from a URL to a temporary location.
    
    Args:
        url: URL to download file from
        
    Returns:
        Path to downloaded file if successful, None otherwise
    """
    try:
        local_filename = os.path.join(tempfile.gettempdir(), url.split('/')[-1])
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return local_filename
    except Exception as e:
        print(f"Error downloading file: {e}")
        return None

@api_app.post("/api/rig-from-url/joints-only")
async def rig_joints_only(request: Request):
    """API endpoint to rig a model with joints only (no animation)
    
    Args:
        request: The FastAPI request with {"url": "model_url", "format": "glb|fbx"}
                 format is optional and defaults to "glb" for PlayCanvas compatibility
    
    Returns:
        JSON response with the URL to the rigged model
    """
    try:
        # Get data from the request
        data = await request.json()
        model_url = data["url"]
        
        # Allow client to request specific format (default to glb for PlayCanvas)
        requested_format = data.get("format", "glb").lower()
        
        print(f"Rigging model from URL: {model_url} (format: {requested_format})")
        
        # Download the model
        local_filename = download_file(model_url)
        if not local_filename:
            return {"status": "error", "message": "Failed to download model from URL"}
        
        # Rig the model with JOINTS ONLY (no animation)
        db = _DB_class()
        for step in _pipeline_function(
            input_path=local_filename,
            is_gs=False,
            opacity_threshold=0.01,
            no_fingers=True,  # From image: "No Fingers" is checked
            rest_pose_type="T-pose",  # From image: T-pose
            ignore_pose_parts=[],  # From image: all parts are checked, which means don't ignore
            input_normal=False,
            bw_fix=True,
            bw_vis_bone="LeftArm",
            reset_to_rest=True,  # From image: checked
            animation_file=None,  # No animation for joints-only
            retarget_animation=True,  # From image: "Retarget Animation" is checked
            in_place=True,  # From image: "In Place" is checked
            export_format=requested_format  # Honor requested format (glb or fbx)
        ):
            print(f"Rigging step: {step}")
        
        # Determine which output file to use - prefer GLB for PlayCanvas
        rigged_path = None
        format_type = requested_format
        
        if requested_format == "glb":
            # For GLB output
            if db.anim_vis_path and os.path.isfile(db.anim_vis_path):
                rigged_path = db.anim_vis_path
            elif db.anim_path and os.path.isfile(db.anim_path):
                # If no GLB but FBX exists, convert it
                rigged_path = convert_fbx_to_glb(db.anim_path)
                if not rigged_path:
                    return {"status": "error", "message": "Failed to convert FBX to GLB"}
        else:
            # For FBX output
            if db.anim_path and os.path.isfile(db.anim_path):
                rigged_path = db.anim_path
            
        if not rigged_path:
            return {"status": "error", "message": "Rigging failed: output file not found"}
        
        # Upload the rigged model to the render.com server
        server_url = upload_to_render_server(rigged_path, model_url)
        if not server_url:
            return {"status": "error", "message": "Failed to upload rigged model to server"}
        
        return {
            "status": "success", 
            "message": "Model rigged successfully",
            "rigged_url": server_url,
            "format": format_type
        }
        
    except Exception as e:
        print(f"Error rigging model: {e}")
        return {"status": "error", "message": f"Failed to rig model: {str(e)}"}

@api_app.post("/api/rig-from-url/animated")
async def rig_animated(request: Request):
    """API endpoint to rig a model WITH ANIMATION
    
    Args:
        request: The FastAPI request with {"url": "model_url", "format": "glb|fbx", "animation_url": "Standard_Run.fbx"}
                 format is optional and defaults to "glb" for PlayCanvas compatibility
                 animation_url is optional and defaults to None
    
    Returns:
        JSON response with the URL to the rigged & animated model
    """
    try:
        # Get data from the request
        data = await request.json()
        model_url = data["url"]
        
        # Allow client to request specific format (default to glb for PlayCanvas)
        requested_format = data.get("format", "glb").lower()
        
        # Allow client to provide an animation file URL (optional)
        animation_url = data.get("animation_url", None)
        animation_path = None
        
        print(f"Rigging & animating model from URL: {model_url} (format: {requested_format})")
        
        # Download the model
        local_filename = download_file(model_url)
        if not local_filename:
            return {"status": "error", "message": "Failed to download model from URL"}
        
        # Download animation file if provided
        if animation_url:
            animation_path = download_file(animation_url)
            if not animation_path:
                return {"status": "error", "message": "Failed to download animation file from URL"}
        
        # Rig and animate the model
        db = _DB_class()
        for step in _pipeline_function(
            input_path=local_filename,
            is_gs=False,
            opacity_threshold=0.01,
            no_fingers=True,  # From image: "No Fingers" is checked
            rest_pose_type="T-pose",  # From image: T-pose
            ignore_pose_parts=[],  # From image: all parts are checked, which means don't ignore
            input_normal=False,
            bw_fix=True,
            bw_vis_bone="LeftArm",
            reset_to_rest=True,  # From image: checked
            animation_file=animation_path,  # Use downloaded animation if provided
            retarget_animation=True,  # From image: "Retarget Animation" is checked
            in_place=True,  # From image: "In Place" is checked
            export_format=requested_format  # Honor requested format (glb or fbx)
        ):
            print(f"Rigging & animation step: {step}")
        
        # Determine which output file to use - prefer GLB for PlayCanvas
        rigged_path = None
        format_type = requested_format
        
        if requested_format == "glb":
            # For GLB output
            if db.anim_vis_path and os.path.isfile(db.anim_vis_path):
                rigged_path = db.anim_vis_path
            elif db.anim_path and os.path.isfile(db.anim_path):
                # If no GLB but FBX exists, convert it
                rigged_path = convert_fbx_to_glb(db.anim_path)
                if not rigged_path:
                    return {"status": "error", "message": "Failed to convert FBX to GLB"}
        else:
            # For FBX output
            if db.anim_path and os.path.isfile(db.anim_path):
                rigged_path = db.anim_path
            
        if not rigged_path:
            return {"status": "error", "message": "Rigging failed: output file not found"}
        
        # Upload the rigged model to the render.com server
        server_url = upload_to_render_server(rigged_path, model_url)
        if not server_url:
            return {"status": "error", "message": "Failed to upload rigged model to server"}
        
        return {
            "status": "success", 
            "message": "Model rigged and animated successfully",
            "rigged_url": server_url,
            "format": format_type
        }
        
    except Exception as e:
        print(f"Error rigging model: {e}")
        return {"status": "error", "message": f"Failed to rig model: {str(e)}"}

@api_app.get("/")
async def root():
    """Root endpoint for the API"""
    return {"message": "Make-It-Animatable API is running"}

def test_rig():
    """Test function for the Rig from URL button in Gradio
    This just downloads a test model and sends it to the rigging function
    """
    # Test model URL
    test_url = "http://ssh.ohio.render.com/public/models/meshy/test_avatar/punk_test.glb"
    local_filename = download_file(test_url)
    
    # Use the pipeline function
    db = _DB_class()
    for step in _pipeline_function(
        input_path=local_filename,
        is_gs=False,
        opacity_threshold=0.01,
        no_fingers=True,
        rest_pose_type="T-pose",
        ignore_pose_parts=[],
        input_normal=False,
        bw_fix=True,
        bw_vis_bone="LeftArm",
        reset_to_rest=True,
        animation_file=None,
        retarget_animation=True,
        in_place=True,
        export_format="glb"
    ):
        print(f"Test rigging step: {step}")
    
    # Determine which output file to use - prefer GLB for PlayCanvas
    if db.anim_vis_path and os.path.isfile(db.anim_vis_path):
        rigged_path = db.anim_vis_path
    elif db.anim_path and os.path.isfile(db.anim_path):
        # If no GLB but FBX exists, convert it
        rigged_path = convert_fbx_to_glb(db.anim_path)
    else:
        print("Test rigging failed: output file not found")
        return None
    
    # Upload the rigged model to the render.com server
    server_url = upload_to_render_server(rigged_path, test_url)
    if not server_url:
        print("Failed to upload test rigged model to server")
        return None
    
    print(f"Test rigged model accessible at: {server_url}")
    return server_url 
