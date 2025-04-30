import os
import tempfile
import threading
import requests
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

@api_app.post("/api/rig-from-url")
async def rig_from_url_api(request: Request):
    """API endpoint to rig a model from a URL
    
    Args:
        request: The FastAPI request
    
    Returns:
        JSON response with the URL to the rigged model
    """
    try:
        # Get data from the request
        data = await request.json()
        model_url = data["url"]
        
        # Get the pipeline function and DB class from the app state
        pipeline_function = api_app.state.pipeline_function
        DB_class = api_app.state.DB_class
        
        # Download the model
        local_filename = os.path.join(tempfile.gettempdir(), model_url.split('/')[-1])
        with requests.get(model_url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        # Rig the model
        db = DB_class()
        for step in pipeline_function(
            input_path=local_filename,
            is_gs=False,
            opacity_threshold=0.01,
            no_fingers=True,
            rest_pose_type="No",
            ignore_pose_parts=[],
            input_normal=False,
            bw_fix=True,
            bw_vis_bone="LeftArm",
            reset_to_rest=True,
            animation_file=None,
            retarget=True,
            inplace=True,
            db=db,
            export_temp=True,
        ):
            pass
            
        # Determine which output file to use
        if db.anim_vis_path and os.path.isfile(db.anim_vis_path):
            rigged_path = db.anim_vis_path
        elif db.anim_path and os.path.isfile(db.anim_path):
            rigged_path = db.anim_path
        else:
            return {"status": "error", "message": "Rigging failed: output file not found"}
        
        # Upload to Render.com
        with open(rigged_path, "rb") as f:
            files = {"modelFile": f}
            response = requests.post("https://viverse-backend.onrender.com/api/upload-rigged-model", files=files)
            if response.status_code != 200:
                return {"status": "error", "message": f"Upload to Render.com failed with status code {response.status_code}"}
            persistent_url = response.json().get("persistentUrl")
            
        return {"status": "done", "persistentUrl": persistent_url}
    except Exception as e:
        return {"status": "error", "message": str(e)} 