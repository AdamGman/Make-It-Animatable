import spaces  # isort:skip
import contextlib
import gc
import os
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass
from glob import glob

# --- ADDED Import ---
import requests # Ensure requests is in requirements.txt

import gradio as gr
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from pytorch3d.transforms import Transform3d

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from model import PCAE
from util.dataset_mixamo import (
    BONES_IDX_DICT,
    JOINTS_NUM,
    KINEMATIC_TREE,
    MIXAMO_PREFIX,
    TEMPLATE_PATH,
    Joint,
    get_hips_transform,
)
from util.dataset_mixamo_additional import BONES_IDX_DICT as BONES_IDX_DICT_ADD
from util.dataset_mixamo_additional import JOINTS_NUM as JOINTS_NUM_ADD
from util.dataset_mixamo_additional import KINEMATIC_TREE as KINEMATIC_TREE_ADD
from util.dataset_mixamo_additional import TEMPLATE_PATH as TEMPLATE_PATH_ADD
from util.utils import (
    TimePrints,
    Timing,
    apply_transform,
    fix_random,
    get_normalize_transform,
    load_gs,
    make_archive, # Make sure this is defined in util.utils
    pose_local_to_global,
    pose_rot_to_global,
    sample_mesh,
    save_gs,
    str2bool,
    str2list,
    to_pose_local,
    to_pose_matrix,
    transform_gs,
)

# ... (Keep all the monkey patching, dataclass DB, helper functions like clear, get_conflict_mask, bw_post_process, etc. from the previous version) ...
# ... (Keep prepare_input, preprocess, infer, vis, model loading functions, etc.) ...

# --- MODIFIED vis_blender function ---
def vis_blender(
    reset_to_rest: bool,
    remove_fingers: bool,
    rest_pose_type: str,
    ignore_pose_parts: list[str],
    animation_file: str,
    retarget: bool,
    inplace: bool,
    db: DB,
):
    # Check if necessary data exists from previous steps
    required_data = (db.joints, db.joints_tail, db.bw)
    input_source = db.mesh if db.is_mesh else db.gs
    if any(x is None for x in required_data) or input_source is None:
        raise gr.Error("Run the inference first (prepare, preprocess, infer, vis steps must complete).")

    # Handle Gaussian Splats warning and numpy conversion
    if not db.is_mesh and db.gs is not None:
        gr.Warning("It can take quite a long time to import and rig Gaussian Splats in Blender. Please wait patiently.")
        if isinstance(db.gs, torch.Tensor):
            db.gs = db.gs.numpy()
        if db.gs_rest is not None and isinstance(db.gs_rest, torch.Tensor):
            db.gs_rest = db.gs_rest.numpy()

    # Determine template path based on model configuration (ADDITIONAL_BONES flag)
    # Ensure joints_additional flag is correctly set during init_models if needed
    template_path = TEMPLATE_PATH_ADD if 'joints_additional' in globals() and joints_additional else TEMPLATE_PATH

    # Prepare data dictionary for app_blender
    data = dict(
        mesh=db.mesh, # Can be None if input is GS
        gs=db.gs_rest if reset_to_rest else db.gs, # Can be None if input is mesh
        joints=db.joints,
        joints_tail=db.joints_tail,
        bw=db.bw,
        pose=db.pose, # Can be None if no pose prediction happened
        bones_idx_dict=dict(bones_idx_dict_joints), # Ensure bones_idx_dict_joints is defined
        pose_ignore_list=get_pose_ignore_list(rest_pose_type, ignore_pose_parts),
    )

    # Validate animation file path if provided
    if animation_file is not None:
        if not isinstance(animation_file, str) or not os.path.isfile(animation_file):
             # Gradio File component might return a tempfile object path
             # Check if it has a 'name' attribute for the path
             if hasattr(animation_file, 'name') and os.path.isfile(animation_file.name):
                 animation_file_path = animation_file.name
             else:
                  raise gr.Error(f"Animation file path is invalid or file does not exist: {animation_file}")
        else:
             animation_file_path = animation_file

        if not reset_to_rest:
            gr.Warning(
                "\'Reset to Rest\' is not enabled, so the animation may be incorrect if the input is not in T-pose"
            )
    else:
        animation_file_path = None # Ensure it's None if no file provided

    # --- Execute Blender Script ---
    # Determine absolute paths needed for the command line execution
    abs_anim_path = os.path.abspath(db.anim_path) if db.anim_path else None
    abs_template_path = os.path.abspath(template_path)
    abs_rest_vis_path = os.path.abspath(db.rest_vis_path) if db.is_mesh and db.rest_vis_path else None
    abs_animation_file_path = os.path.abspath(animation_file_path) if animation_file_path else None

    if is_main_thread():
        from argparse import Namespace
        from app_blender import main # Assumes app_blender.py is in the same directory or sys.path

        print("[vis_blender] Executing app_blender.py in main thread...")
        main(
            Namespace(
                input_path=data, # Pass data dict directly
                output_path=abs_anim_path, # Use absolute path
                template_path=abs_template_path,
                keep_raw=False,
                rest_path=abs_rest_vis_path,
                pose_local=False,
                reset_to_rest=reset_to_rest,
                remove_fingers=remove_fingers,
                animation_path=abs_animation_file_path,
                retarget=retarget,
                inplace=inplace,
            )
        )
    else:
        print("[vis_blender] Executing app_blender.py in child thread via os.system...")
        with tempfile.NamedTemporaryFile(suffix=".npz") as f:
            np.savez(f.name, **data) # Save data to temp file
            cmd = f"python app_blender.py --input_path '{f.name}' --output_path '{abs_anim_path}'"
            cmd += f" --template_path '{abs_template_path}'"
            if abs_rest_vis_path: # Check if path exists
                cmd += f" --rest_path '{abs_rest_vis_path}'"
            if reset_to_rest:
                cmd += " --reset_to_rest"
            if remove_fingers:
                cmd += " --remove_fingers"
            if abs_animation_file_path: # Check if path exists
                cmd += f" --animation_path '{abs_animation_file_path}'"
                if retarget:
                    cmd += " --retarget"
                if inplace:
                    cmd += " --inplace"
            # Redirect output; consider removing redirection if debugging blender script
            cmd += " > /dev/null 2>&1"
            # print(f"[vis_blender] Running command: {cmd}") # Uncomment for debugging
            exit_code = os.system(cmd)
            if exit_code != 0:
                 print(f"ERROR: app_blender.py execution failed with exit code {exit_code}.")
                 # Optionally raise an error or set a flag
            else:
                 print("[vis_blender] app_blender.py execution completed.")

    # --- Check if output exists and Upload to Render.com ---
    print(f"Output animatable model potentially generated at temporary path: '{db.anim_path}'")

    persistent_url = None
    # Default to the temporary path generated by app_blender
    final_output_path_or_url = db.anim_path if db.anim_path and os.path.isfile(db.anim_path) else None

    # Check if the output file was actually created by app_blender.py
    if final_output_path_or_url: # Check if path is valid and file exists

        anim_path_to_upload = final_output_path_or_url # Path to the file to potentially upload

        # Optional: Handle Compression
        try:
             file_size_mb = os.path.getsize(anim_path_to_upload) / (1024**2)
             if file_size_mb > 50: # Example threshold 50MB
                 gr.Info(f"Animation file is large ({file_size_mb:.2f}MB), compressing before upload...")
                 compressed_path = f"{os.path.splitext(anim_path_to_upload)[0]}.zip"
                 make_archive(anim_path_to_upload, compressed_path) # Creates the zip
                 anim_path_to_upload = compressed_path # Update path to the zip file
                 print(f"Compressed file for upload: '{anim_path_to_upload}'")
        except Exception as comp_e:
             print(f"WARNING: Failed to check size or compress file {anim_path_to_upload}: {comp_e}")
             # Continue with uncompressed file if compression fails or size check fails

        # --- Add Upload Logic to Render.com ---
        try:
            # *** USE THE CORRECT RENDER.COM ENDPOINT URL ***
            render_upload_endpoint = "https://viverse-backend.onrender.com/api/upload-rigged-model"
            # Add headers if your Render endpoint needs authentication
            # headers = {'Authorization': f'Bearer {os.environ.get("RENDER_API_SECRET")}'}
            headers = {} # No auth headers assumed for now

            # Use 'with open' for safer file handling
            with open(anim_path_to_upload, 'rb') as f_upload:
                 # 'modelFile' must match the key expected by multer on Render.com server
                 file_to_upload = {'modelFile': (os.path.basename(anim_path_to_upload), f_upload)}

                 print(f"Uploading {anim_path_to_upload} to Render.com endpoint: {render_upload_endpoint}...")

                 response = requests.post(render_upload_endpoint, files=file_to_upload, headers=headers)
                 response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

                 response_data = response.json()
                 if response_data.get("persistentUrl"):
                     persistent_url = response_data["persistentUrl"]
                     final_output_path_or_url = persistent_url # IMPORTANT: Update to the returned persistent URL
                     print(f"Successfully uploaded to Render.com. Persistent URL: {persistent_url}")
                 else:
                     print("ERROR: Render.com upload response did not contain 'persistentUrl'. Will return temporary path.")
                     # final_output_path_or_url remains the local path

        except requests.exceptions.RequestException as e:
            print(f"ERROR: Failed to upload to Render.com: {e}. Will return temporary path.")
            # final_output_path_or_url remains the local path
        except Exception as e:
            print(f"ERROR: An unexpected error occurred during Render.com upload: {e}. Will return temporary path.")
            # final_output_path_or_url remains the local path
        # --- End Upload Logic ---

    else: # File db.anim_path was not found
        print(f"ERROR: Output file {db.anim_path} not found after app_blender execution. Cannot upload or return.")
        final_output_path_or_url = None # Ensure it's None if file wasn't created

    # --- Post-processing for FBX to GLB visualization ---
    # This uses the *local* db.anim_path (before potential compression/URL change)
    if db.is_mesh and db.anim_path and db.anim_path.endswith(".fbx") and os.path.isfile(db.anim_path):
        with tempfile.TemporaryDirectory() as tmpdir:
            fbx2gltf_path = "util/FBX2glTF"
            if not os.path.exists(fbx2gltf_path):
                 fbx2gltf_path = "FBX2glTF"
            if os.popen(f"command -v {fbx2gltf_path}").read().strip():
                 fbx2glb_cmd = f"{fbx2gltf_path} --binary --keep-attribute auto --fbx-temp-dir '{tmpdir}' --input '{os.path.abspath(db.anim_path)}' --output '{os.path.abspath(db.anim_vis_path)}'"
                 fbx2glb_cmd += " > /dev/null 2>&1"
                 os.system(fbx2glb_cmd)
                 if os.path.isfile(db.anim_vis_path):
                      print(f"Output visualization (GLB for UI): '{db.anim_vis_path}'")
                 else:
                      print(f"ERROR: FBX to GLB conversion failed.")
                      db.anim_vis_path = None
            else:
                 print(f"ERROR: FBX2glTF command ('{fbx2gltf_path}') not found.")
                 db.anim_vis_path = None
    else:
        db.anim_vis_path = None # No GLB vis if input wasn't mesh or output wasn't FBX


    # --- Return results for Gradio UI ---
    return {
        output_rest_vis: db.rest_vis_path,          # Local path for UI preview (GLB)
        output_anim: final_output_path_or_url,      # Render.com URL or local temp path/None
        output_anim_vis: db.anim_vis_path,          # Local path for UI preview (GLB)
        state: db, # Pass the state back, might contain useful info
    }

# ... (Keep finish, _pipeline, init_models, init_blocks functions) ...
# ... (Ensure _pipeline yields the result from vis_blender correctly) ...
# ... (Ensure init_blocks maps the output components correctly, especially output_anim to receive the URL/path) ...

if __name__ == "__main__":
    # Ensure models are loaded relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Changed working directory to: {script_dir}")

    init_models() # Load models once on startup
    demo = init_blocks() # Setup Gradio interface

    # Launch the Gradio app
    print("Launching Gradio App...")
    demo.queue(default_concurrency_limit=1).launch( # Use queue for handling requests
         server_name="0.0.0.0", # Allow external connections
         server_port=7860, # Default Gradio port
         # allowed_paths=["*"], # Consider security implications
         show_error=True, # Show errors in browser console
         # ssl_verify=False # Use if running locally without proper SSL
    )
