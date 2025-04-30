import os
import tempfile
import threading
import requests
import re
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
    # We need to pass these to the API endpoints
    api_app.state.pipeline_function = pipeline_function
    api_app.state.DB_class = DB_class
    
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
    """Extract the base filename from a URL"""
    # Get just the filename without extension
    filename = os.path.basename(url)
    base_name = os.path.splitext(filename)[0]
    return base_name

def upload_to_render(file_path, original_url=None):
    """Upload a file to Render.com
    
    Args:
        file_path: Local path to file to upload
        original_url: Original URL the model came from (to determine upload location)
        
    Returns:
        URL of the uploaded file
    """
    # Extract the folder path from the original URL if available
    upload_url = "https://viverse-backend.onrender.com/api/upload-rigged-model"
    
    if original_url:
        # Try to extract the folder from the original URL to maintain organization
        folder_match = re.match(r'(.*)/[^/]+$', original_url)
        if folder_match:
            folder = folder_match.group(1)
            # Add folder information to the upload request
            upload_url += f"?folder={folder}"
    
    with open(file_path, "rb") as f:
        files = {"modelFile": f}
        response = requests.post(upload_url, files=files)
        if response.status_code != 200:
            raise Exception(f"Upload to Render.com failed with status code {response.status_code}")
        
        persistent_url = response.json().get("persistentUrl")
        if not persistent_url:
            raise Exception("No persistentUrl returned from Render.com")
        
        return persistent_url

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
        
        # Allow client to request specific format, but default to GLB for PlayCanvas
        preferred_format = data.get("format", "glb").lower()
        
        # Get the pipeline function and DB class from the app state
        pipeline_function = api_app.state.pipeline_function
        DB_class = api_app.state.DB_class
        
        # Extract the original filename for consistent naming
        base_name = extract_filename_from_url(model_url)
        
        # Download the model
        local_filename = os.path.join(tempfile.gettempdir(), os.path.basename(model_url))
        with requests.get(model_url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        # Rig the model with JOINTS ONLY (no animation)
        db = DB_class()
        for step in pipeline_function(
            input_path=local_filename,
            is_gs=False,
            opacity_threshold=0.01,
            no_fingers=False,  # Default: keep fingers
            rest_pose_type="T-pose",  # From image: T-pose
            ignore_pose_parts=["Fingers", "Arms", "Legs", "Head"],  # From image: all parts checked
            input_normal=False,
            bw_fix=True,
            bw_vis_bone="LeftArm",
            reset_to_rest=True,  # From image: checked
            animation_file=None,  # No animation for joints-only
            retarget=True,
            inplace=True,
            db=db,
            export_temp=True,
        ):
            pass
            
        # Determine which output file to use based on preferred format and availability
        if preferred_format == "glb" and db.anim_vis_path and os.path.isfile(db.anim_vis_path):
            # GLB is preferred for PlayCanvas
            rigged_path = db.anim_vis_path
            format_type = "glb"
        elif preferred_format == "fbx" and db.anim_path and os.path.isfile(db.anim_path):
            # FBX if specifically requested
            rigged_path = db.anim_path
            format_type = "fbx" 
        elif db.anim_vis_path and os.path.isfile(db.anim_vis_path):
            # Fallback to GLB if available regardless of preference
            rigged_path = db.anim_vis_path
            format_type = "glb"
        elif db.anim_path and os.path.isfile(db.anim_path):
            # Last resort: FBX
            rigged_path = db.anim_path
            format_type = "fbx"
        else:
            return {"status": "error", "message": "Rigging failed: output file not found"}

        # Rename the file with _joints_only suffix
        extension = os.path.splitext(rigged_path)[1]
        renamed_path = os.path.join(os.path.dirname(rigged_path), f"{base_name}_joints_only{extension}")
        os.rename(rigged_path, renamed_path)
        
        # Upload to Render.com
        persistent_url = upload_to_render(renamed_path, model_url)
            
        return {
            "status": "done", 
            "persistentUrl": persistent_url,
            "modelType": "joints_only",
            "format": format_type
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@api_app.post("/api/rig-from-url/animated")
async def rig_animated(request: Request):
    """API endpoint to rig a model with Standard Run animation
    
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
        
        # Allow client to request specific format, but default to GLB for PlayCanvas
        preferred_format = data.get("format", "glb").lower()
        
        # Get the pipeline function and DB class from the app state
        pipeline_function = api_app.state.pipeline_function
        DB_class = api_app.state.DB_class
        
        # Extract the original filename for consistent naming
        base_name = extract_filename_from_url(model_url)
        
        # Download the model
        local_filename = os.path.join(tempfile.gettempdir(), os.path.basename(model_url))
        with requests.get(model_url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        # Get animation file path
        animation_path = get_animation_path()
        
        # Rig the model with ANIMATION
        db = DB_class()
        for step in pipeline_function(
            input_path=local_filename,
            is_gs=False,
            opacity_threshold=0.01,
            no_fingers=False,  # Default: keep fingers
            rest_pose_type="T-pose",  # From image: T-pose
            ignore_pose_parts=["Fingers", "Arms", "Legs", "Head"],  # From image: all parts checked
            input_normal=False,
            bw_fix=True,
            bw_vis_bone="LeftArm",
            reset_to_rest=True,  # From image: checked
            animation_file=animation_path,  # Use Standard Run animation
            retarget=True,  # From image: checked
            inplace=True,  # From image: checked
            db=db,
            export_temp=True,
        ):
            pass
            
        # Determine which output file to use based on preferred format and availability
        if preferred_format == "glb" and db.anim_vis_path and os.path.isfile(db.anim_vis_path):
            # GLB is preferred for PlayCanvas
            rigged_path = db.anim_vis_path
            format_type = "glb"
        elif preferred_format == "fbx" and db.anim_path and os.path.isfile(db.anim_path):
            # FBX if specifically requested
            rigged_path = db.anim_path
            format_type = "fbx" 
        elif db.anim_vis_path and os.path.isfile(db.anim_vis_path):
            # Fallback to GLB if available regardless of preference
            rigged_path = db.anim_vis_path
            format_type = "glb"
        elif db.anim_path and os.path.isfile(db.anim_path):
            # Last resort: FBX
            rigged_path = db.anim_path
            format_type = "fbx"
        else:
            return {"status": "error", "message": "Rigging failed: output file not found"}

        # Rename the file with _animated suffix
        extension = os.path.splitext(rigged_path)[1]
        renamed_path = os.path.join(os.path.dirname(rigged_path), f"{base_name}_animated{extension}")
        os.rename(rigged_path, renamed_path)
        
        # Upload to Render.com
        persistent_url = upload_to_render(renamed_path, model_url)
            
        return {
            "status": "done", 
            "persistentUrl": persistent_url,
            "modelType": "animated",
            "animation": "Standard Run",
            "format": format_type
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Keep the original endpoint for backward compatibility
@api_app.post("/api/rig-from-url")
async def rig_from_url_api(request: Request):
    """Legacy API endpoint to rig a model (uses animated version)"""
    return await rig_animated(request) 
