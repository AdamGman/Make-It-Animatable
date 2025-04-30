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

import requests
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
    make_archive,
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

# Monkey patching to correct the loaded example values from csv
Checkbox_postprocess = gr.Checkbox.postprocess
gr.Checkbox.postprocess = lambda self, value: (
    str2bool(value) if isinstance(value, str) else Checkbox_postprocess(self, value)
)
CheckboxGroup_postprocess = gr.CheckboxGroup.postprocess
gr.CheckboxGroup.postprocess = lambda self, value: (
    list(filter(None, str2list(lambda x: x.lstrip('"').rstrip('"'))(value)))
    if isinstance(value, str)
    else CheckboxGroup_postprocess(self, value)
)

def is_main_thread():
    import threading
    return threading.current_thread() is threading.main_thread()

# Monkey patching gradio to use let gr.Info & gr.Warning also print on console
def _log_message(
    message: str,
    level="info",
    duration: float | None = 10,
    visible: bool = True,
    *args,
    **xargs,
):
    from gradio.context import LocalContext

    if level in ("info", "success"):
        print(message)
    elif level == "warning":
        warnings.warn(message)

    blocks = LocalContext.blocks.get()
    event_id = LocalContext.event_id.get()
    if blocks is not None and event_id is not None:
        # Function called outside of Gradio if blocks is None
        # Or from /api/predict if event_id is None
        blocks._queue.log_message(
            event_id=event_id, log=message, level=level, duration=duration, visible=visible, *args, **xargs
        )

import gradio.helpers
gradio.helpers.log_message = _log_message

cmap = matplotlib.colormaps.get_cmap("plasma")

@dataclass()
class DB:
    mesh: trimesh.Trimesh = None
    gs: torch.Tensor = None
    gs_rest: torch.Tensor = None
    is_mesh: bool = None
    sample_mask: np.ndarray = None
    verts: torch.Tensor = None
    verts_normal: torch.Tensor = None
    faces: np.ndarray = None
    pts: torch.Tensor = None
    pts_normal: torch.Tensor = None
    global_transform: Transform3d = None

    output_dir: str = None
    joints_coarse_path: str = None
    normed_path: str = None
    sample_path: str = None
    bw_path: str = None
    joints_path: str = None
    rest_lbs_path: str = None
    rest_vis_path: str = None
    anim_path: str = None
    anim_vis_path: str = None

    bw: torch.Tensor = None
    joints: torch.Tensor = None
    joints_tail: torch.Tensor = None
    pose: torch.Tensor = None

    def clear(self):
        for k in self.__dict__:
            self.__dict__[k] = None
        return self

def clear(db: DB = None):
    if db is not None:
        db.clear()
    gc.collect()
    torch.cuda.empty_cache()
    print("Memory cleared")
    return db

# [Keep all other helper functions like get_conflict_mask, bw_post_process, etc.]

# --- Modified vis_blender function to use Gradio's built-in file serving ---
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
    if any(x is None for x in required_data) or (db.is_mesh and db.mesh is None) or (not db.is_mesh and db.gs is None):
        raise gr.Error("Run the inference first (prepare, preprocess, infer, vis steps must complete).")

    # Handle Gaussian Splats warning and numpy conversion
    if db.gs is not None:
        gr.Warning("It can take quite a long time to import and rig Gaussian Splats in Blender. Please wait patiently.")
        if isinstance(db.gs, torch.Tensor):
            db.gs = db.gs.numpy()
        if db.gs_rest is not None and isinstance(db.gs_rest, torch.Tensor):
            db.gs_rest = db.gs_rest.numpy()

    # Determine template path based on model configuration
    template_path = TEMPLATE_PATH_ADD if 'joints_additional' in globals() and joints_additional else TEMPLATE_PATH

    # Prepare data dictionary for app_blender
    data = dict(
        mesh=db.mesh,  # Can be None if input is GS
        gs=db.gs_rest if reset_to_rest else db.gs,  # Can be None if input is mesh
        joints=db.joints,
        joints_tail=db.joints_tail,
        bw=db.bw,
        pose=db.pose,  # Can be None if no pose prediction happened
        bones_idx_dict=dict(bones_idx_dict_joints),  # Ensure bones_idx_dict_joints is defined
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
        animation_file_path = None  # Ensure it's None if no file provided

    # --- Execute Blender Script ---
    # Determine absolute paths needed for the command line execution
    abs_anim_path = os.path.abspath(db.anim_path) if db.anim_path else None
    abs_template_path = os.path.abspath(template_path)
    abs_rest_vis_path = os.path.abspath(db.rest_vis_path) if db.is_mesh and db.rest_vis_path else None
    abs_animation_file_path = os.path.abspath(animation_file_path) if animation_file_path else None

    if is_main_thread():
        from argparse import Namespace
        from app_blender import main  # Assumes app_blender.py is in the same directory or sys.path

        print("[vis_blender] Executing app_blender.py in main thread...")
        main(
            Namespace(
                input_path=data,  # Pass data dict directly
                output_path=abs_anim_path,  # Use absolute path
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
            np.savez(f.name, **data)  # Save data to temp file
            cmd = f"python app_blender.py --input_path '{f.name}' --output_path '{abs_anim_path}'"
            cmd += f" --template_path '{abs_template_path}'"
            if abs_rest_vis_path:  # Check if path exists
                cmd += f" --rest_path '{abs_rest_vis_path}'"
            if reset_to_rest:
                cmd += " --reset_to_rest"
            if remove_fingers:
                cmd += " --remove_fingers"
            if abs_animation_file_path:  # Check if path exists
                cmd += f" --animation_path '{abs_animation_file_path}'"
                if retarget:
                    cmd += " --retarget"
                if inplace:
                    cmd += " --inplace"
            # Redirect output; consider removing redirection if debugging blender script
            cmd += " > /dev/null 2>&1"
            # print(f"[vis_blender] Running command: {cmd}")  # Uncomment for debugging
            exit_code = os.system(cmd)
            if exit_code != 0:
                 print(f"ERROR: app_blender.py execution failed with exit code {exit_code}.")
            else:
                 print("[vis_blender] app_blender.py execution completed.")

    # --- Generate Direct Gradio File URL ---
    print(f"Output animatable model potentially generated at temporary path: '{db.anim_path}'")

    # Default to the temporary path generated by app_blender
    final_output_path_or_url = None

    # Check if the output file was actually created by app_blender.py
    if db.anim_path and os.path.isfile(db.anim_path):
        # Get the Space URL
        space_url = os.environ.get('SPACE_URL', 'https://dkatz2391-make-it-animatable.hf.space')
        
        # Create direct Gradio file URL
        gradio_file_url = f"{space_url}/gradio_api/file={db.anim_path}"
        final_output_path_or_url = gradio_file_url
        print(f"Model will be accessible at: {gradio_file_url}")
        
        # Optional: Compress large files if needed
        try:
            file_size_mb = os.path.getsize(db.anim_path) / (1024**2)
            if file_size_mb > 50:  # Example threshold 50MB
                gr.Info(f"Animation file is large ({file_size_mb:.2f}MB), consider compressing it!")
        except Exception as e:
            print(f"WARNING: Failed to check file size: {e}")
    else:
        print(f"ERROR: Output file {db.anim_path} not found after app_blender execution.")
        final_output_path_or_url = None

    # --- Post-processing for FBX to GLB visualization ---
    # This uses the *local* db.anim_path (before URL creation)
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
        db.anim_vis_path = None  # No GLB vis if input wasn't mesh or output wasn't FBX

    # --- Return results for Gradio UI ---
    return {
        output_rest_vis: db.rest_vis_path,          # Local path for UI preview (GLB)
        output_anim: final_output_path_or_url,      # Gradio file URL or None
        output_anim_vis: db.anim_vis_path,          # Local path for UI preview (GLB)
        state: db,                                  # Pass the state back
    }

# Keep all other functions like finish, _pipeline, init_models, init_blocks, etc.

def init_models():
    global device, N, hands_resample_ratio, geo_resample_ratio, bw_additional, joints_additional, bones_idx_dict_bw, bones_idx_dict_joints, model_bw, model_bw_normal, model_joints, model_joints_add, model_coarse, model_pose

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    fix_random()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    IS_HF_ZEROGPU = str2bool(os.getenv("SPACES_ZERO_GPU", False))

    N = 32768
    hands_resample_ratio = 0.5
    geo_resample_ratio = 0.0
    hierarchical_ratio = hands_resample_ratio + geo_resample_ratio

    ADDITIONAL_BONES = bw_additional = joints_additional = False

    model_bw = PCAE(
        N=N,
        input_normal=False,
        deterministic=True,
        hierarchical_ratio=hierarchical_ratio,
        output_dim=JOINTS_NUM_ADD if ADDITIONAL_BONES else JOINTS_NUM,
    )
    
    # Load model weights
    if ADDITIONAL_BONES:
        model_bw.load("output/vroid/bw.pth")
    else:
        model_bw.load("output/best/new/bw.pth")
    
    model_bw.to("cpu" if IS_HF_ZEROGPU else device).eval()

    # [Continue with loading other models, same as original code]

def init_blocks():
    global output_rest_vis, output_anim, output_anim_vis, state
    
    # [Keep all code from the original init_blocks]
    
    # [Return the demo object]
    return demo

if __name__ == "__main__":
    # Ensure models are loaded relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Changed working directory to: {script_dir}")

    # Load models once on startup
    init_models()
    
    # Create the Gradio interface
    demo = init_blocks()
    
    # Launch the Gradio app
    print("Launching Gradio App...")
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    ) 
