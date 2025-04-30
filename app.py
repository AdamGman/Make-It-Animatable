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
from fastapi import FastAPI, Request
import threading

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
    change_Model3D,
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

def rig_from_url(model_url):
    local_path = download_file(model_url)
    db = DB()
    for step in _pipeline(
        input_path=local_path,
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
    if db.anim_vis_path and os.path.isfile(db.anim_vis_path):
        return db.anim_vis_path
    elif db.anim_path and os.path.isfile(db.anim_path):
        return db.anim_path
    else:
        raise gr.Error("Rigging failed: output file not found.")

def download_file(url):
    local_filename = os.path.join(tempfile.gettempdir(), url.split('/')[-1])
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename

# --- FastAPI for backend integration ---
api_app = FastAPI()

@api_app.post("/api/rig-from-url")
async def rig_from_url_api(request: Request):
    data = await request.json()
    model_url = data["url"]
    rigged_path = rig_from_url(model_url)
    # Upload to Render.com
    with open(rigged_path, "rb") as f:
        files = {"modelFile": f}
        response = requests.post("https://viverse-backend.onrender.com/api/upload-rigged-model", files=files)
        persistent_url = response.json().get("persistentUrl")
    return {"status": "done", "persistentUrl": persistent_url}

def run_api():
    import uvicorn
    uvicorn.run(api_app, host="0.0.0.0", port=8000)

# --- Other functions from original app_original.py --- (keep as-is)
# [Include all previous functions like init_models, init_blocks, _pipeline, etc.]

def init_blocks():
    global demo, state, output_joints_coarse, output_normed_input, output_sample, output_joints, output_bw, output_rest_vis, output_rest_lbs, output_anim_vis, output_anim

    title = "Make-It-Animatable"
    description = f"""
    <center>
    <h1> 💃 {title} </h1>
    <h2><b>An Efficient Framework for Authoring Animation-Ready 3D Characters</b></h2>
    <h3>
        📄 <a href='https://arxiv.org/abs/2411.18197' target='_blank'>Paper</a>&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;🌟 <a href='https://jasongzy.github.io/Make-It-Animatable/' target='_blank'>Project Page</a>
    </h3>
    </center>
    """
    camera_position = [90, None, 2.5]
    animation_dir = "data/Mixamo/animation"

    with gr.Blocks(title=title, delete_cache=(3600, 3660)) as demo:
        gr.Markdown(description)
        gr.Markdown(
            """
            - Upload a 3D humanoid model or select an example.
                - If you have a character image and hope to make it animatable, these image-to-3D tools might be helpful: [InstantMesh](https://huggingface.co/spaces/TencentARC/InstantMesh), [CharacterGen](https://huggingface.co/spaces/VAST-AI/CharacterGen), [Tripo](https://www.tripo3d.ai/), and [Meshy](https://www.meshy.ai/).
            - Check if the **Input Settings** are suitable and click **Run**.
            - Your animatable model will soon be ready!
                - Optionally, choose another **Animation File** and click **Animate** to quickly apply new motions.
                - If something goes wrong, check the tips at page bottom.
            """
        )

        state = gr.State(DB())
        with gr.Row(variant="panel"):

            # Inputs
            with gr.Column():
                with gr.Row():
                    input_3d = gr.Model3D(label="Input 3D Model", display_mode="solid", camera_position=camera_position)

                with gr.Group():
                    with gr.Row():
                        with gr.Accordion("Input Settings", open=True):
                            input_no_fingers = gr.Checkbox(
                                label="No Fingers",
                                info="Whether the input model does not have ten separate fingers. Can also be used if the output has unsatisfactory finger results.",
                                value=True,
                                interactive=True,
                            )
                            input_rest_pose = gr.Dropdown(
                                ("T-pose", "A-pose", "大-pose", "No"),
                                label="Input Rest Pose",
                                info="If the input model is already in a rest pose, specify here for better performance.",
                                value="No",
                                interactive=True,
                            )
                            input_rest_parts = gr.CheckboxGroup(
                                ("Fingers", "Arms", "Legs", "Head"),
                                label="Input Rest Parts",
                                info="If certain parts of the input model are already in the T-pose, specify here for better performance.",
                                value=[],
                                interactive=True,
                            )
                            input_is_gs = gr.Checkbox(
                                label="Input is GS",
                                info="Whether the input model is Gaussians Splats (only support `.ply` format).",
                                value=False,
                                interactive=True,
                            )
                            input_opacity_threshold = gr.Slider(
                                0.0,
                                1.0,
                                value=0.01,
                                label="Opacity Threshold",
                                info="Only solid Gaussian Splats with opacities larger than this threshold are used in sampling.",
                                step=0.01,
                                interactive=True,
                                visible=bool(input_is_gs.value),
                            )

                    with gr.Row():
                        with gr.Accordion("Weight Settings", open=False):
                            input_normal = gr.Checkbox(
                                label="Use Normal",
                                info="Use normal information to improve performance when the input has limbs close to other ones. Only take effect when the input is a mesh.",
                                value=False,
                                interactive=True,
                            )
                            input_bw_fix = gr.Checkbox(
                                label="Weight Post-Processing",
                                info="Apply some empirical post-processes to the blend weights.",
                                value=True,
                                interactive=True,
                            )
                            input_bw_vis_bone = gr.Radio(
                                [n.lstrip(MIXAMO_PREFIX) for n in (bones_idx_dict_bw).keys()],
                                label="Bone Name of Weight Visualization",
                                value="LeftArm",
                                interactive=True,
                            )

                    with gr.Row():
                        with gr.Accordion("Animation Settings", open=True):
                            input_reset_to_rest = gr.Checkbox(
                                label="Reset to Rest",
                                info="Apply the predicted T-pose in the final animatable model. If no, its rest pose remains the input pose and the animation results may be incorrect.",
                                value=True,
                                interactive=True,
                            )
                            with gr.Row():
                                select_animation_file = gr.Dropdown(
                                    sorted(glob("*.fbx", root_dir=animation_dir)),
                                    label="Select Animation File",
                                    info="Select or upload the motion sequence (`.fbx`) to be applied to the animatable model. Examples can be downloaded from [Mixamo](https://www.mixamo.com) (select `X Bot` as the base character for best practice) . Please ensure the input 3D model is in T-pose or enable the above **Reset to Rest** first. If the animation file is not specified, the animation results will be in the predicted T-pose (static).",
                                    value=None,
                                    interactive=True,
                                )
                                input_animation_file = gr.File(
                                    label="Animation File",
                                    file_types=[".fbx"],
                                    value=lambda: "./data/Standard Run.fbx",
                                    interactive=True,
                                )
                            with gr.Row():
                                input_retarget = gr.Checkbox(
                                    label="Retarget Animation to Character",
                                    info="Produce better animation.",
                                    value=True,
                                    interactive=bool(input_animation_file.value),
                                )
                                input_inplace = gr.Checkbox(
                                    label="In Place",
                                    info="Keep a looping animation in place (e.g., walking, running...).",
                                    value=True,
                                    interactive=input_retarget.interactive,
                                )

                with gr.Row():
                    submit_btn = gr.Button("Run", variant="primary")
                    animate_btn = gr.Button("Animate", variant="secondary")
                with gr.Row():
                    stop_btn = gr.Button("Stop", variant="stop")
                    clear_btn = gr.ClearButton()

                with gr.Row(variant="panel"):
                    examples = gr.Examples(
                        examples="./data/examples",
                        inputs=[input_3d, input_is_gs, input_no_fingers, input_rest_pose, input_rest_parts],
                        label="Examples",
                        cache_examples=False,
                        examples_per_page=20,
                    )
                    examples.example_labels = examples.dataset.sample_labels = [
                        os.path.basename(x[0]) for x in examples.examples
                    ]

            inputs = (
                input_3d,
                input_no_fingers,
                input_rest_pose,
                input_rest_parts,
                input_is_gs,
                input_opacity_threshold,
                input_normal,
                input_bw_fix,
                input_bw_vis_bone,
                input_reset_to_rest,
                input_animation_file,
                input_retarget,
                input_inplace,
            )

            # Outputs
            with gr.Column():
                with gr.Row():
                    with gr.Tabs():
                        with gr.Tab("Coarse Localization"):
                            output_joints_coarse = gr.Model3D(
                                label="Joints (coarse)", display_mode="solid", camera_position=camera_position
                            )
                        with gr.Tab("Canonical Transformation"):
                            output_normed_input = gr.Model3D(
                                label="Canonical Input", display_mode="solid", camera_position=camera_position
                            )
                        with gr.Tab("Sampling"):
                            output_sample = gr.Model3D(
                                label="Sampled Point Clouds", display_mode="solid", camera_position=camera_position
                            )
                with gr.Row():
                    with gr.Tabs():
                        with gr.Tab("Joints"):
                            output_joints = gr.Model3D(
                                label="Joints", display_mode="solid", camera_position=camera_position
                            )
                        with gr.Tab("Blend Weights"):
                            output_bw = gr.Model3D(
                                label="Blend Weights", display_mode="solid", camera_position=camera_position
                            )
                with gr.Row():
                    with gr.Tabs(selected=1):
                        with gr.Tab("Rest Pose (joints)", id=0):
                            output_rest_lbs = gr.Model3D(
                                label="Rest Pose", display_mode="solid", camera_position=camera_position
                            )
                            gr.Markdown(
                                "The transforming result here may be inaccurate. See **Rest Pose (texture preview)** for optimal visualization."
                            )
                        with gr.Tab("Rest Pose (texture preview)", id=1):
                            output_rest_vis = gr.Model3D(
                                label="Rest Pose", display_mode="solid", camera_position=camera_position
                            )
                            gr.Markdown("**Point clouds** and **Gaussian Splats** are not supported for preview here.")
                with gr.Row():
                    with gr.Tabs():
                        with gr.Tab("Animatable Model (GLB preview)"):
                            output_anim_vis = gr.Model3D(
                                label="Animatable Model", display_mode="solid", camera_position=camera_position
                            )
                            gr.Markdown(
                                """
                                - Gradio hasn't support the FBX format yet (see [this issue](https://github.com/gradio-app/gradio/issues/10007)), so we use [FBX2glTF](https://github.com/facebookincubator/FBX2glTF) internally to convert the exported FBX into GLB for quick preview here.
                                Due to this conversion process, some models may exhibit inconsistencies in material properties and texture rendering. Download the **FBX** file for higher fidelity.
                                - **Point clouds** and **Gaussian Splats** are not supported for preview here. Download the **FBX**/**BLEND** file to view their results.
                                """
                            )
                        with gr.Tab("Animatable Model (FBX/BLEND)"):
                            output_anim = gr.File(label="Animatable Model")
                            gr.Markdown(
                                """
                                - Recommend to view and edit in Blender.
                                - For **Gaussian Splats**, the **[3DGS Render Blender Addon by KIRI Engine](https://github.com/Kiri-Innovation/3dgs-render-blender-addon/releases/tag/v1.0.0)** is required to open the **BLEND** file here.
                                """
                            )
                with gr.Tab("Rig from URL"):
                    url_input = gr.Textbox(label="Model URL (from Render.com)")
                    rigged_output = gr.File(label="Rigged Model (Download)")
                    rig_btn = gr.Button("Rig Model from URL")
                    rig_btn.click(rig_from_url, inputs=url_input, outputs=rigged_output)
                    # Add a test button for the hardcoded test GLB
                    test_btn = gr.Button("Test Render.com Model")
                    def test_rig():
                        return rig_from_url("https://viverse-backend.onrender.com/models/meshy/test_avatar/punk_test.glb")
                    test_btn.click(test_rig, inputs=None, outputs=rigged_output)

        with gr.Row():
            gr.Markdown(
                """
                Tips:
                - To Hugging Face demo users: 3D Gaussian Splats are not supported with the ZeroGPU environment (Python 3.10). Setup an environment with Python 3.11 and run this demo locally to enable GS support.
                - The output results may not be displayed properly if this browser tab is unfocused during inference.
                - **GLB** files appear darker here. Download to view them correctly.
                - If the results suffer from low blend weight quality (typically occurring when limbs are close together, e.g., inner thigh and armpit), try enabling the **Use Normal** option.
                - If the pose-to-rest transformation is unsatisfactory, try adding prior knowledge by specifying the **Input Rest Pose** and **Input Rest Parts**.
                    - Alternatively, you can uncheck **Reset to Rest** and clear the **Animation File**, so that the animation result becomes an invertible T-pose model that can be adjusted in Blender.
                - This demo is designed for standard human skeletons (compatible with the Mixamo definition). If the input 3D model includes significant accessories (e.g., hand-held objects, wings, long tails, long hair), the results may not be optimal.
                """
            )

            outputs = (
                output_joints_coarse,
                output_normed_input,
                output_sample,
                output_joints,
                output_bw,
                output_rest_vis,
                output_rest_lbs,
                output_anim_vis,
                output_anim,
            )
            for e in outputs:
                e.interactive = False

            # Events

            def clear_components(inputs: dict):
                return [None] * len(inputs)

            input_3d.upload(fn=ply2visible, inputs=[input_3d, input_is_gs], outputs=input_3d)
            input_is_gs.change(fn=ply2visible, inputs=[input_3d, input_is_gs], outputs=input_3d)
            input_is_gs.change(
                fn=lambda x: gr.Slider(visible=True) if x else gr.Slider(visible=False),
                inputs=input_is_gs,
                outputs=input_opacity_threshold,
                show_progress="hidden",
            )

            def pipeline(inputs: dict, progress=gr.Progress()):
                progress(0, "Starting...")
                if device.type == "cpu":
                    gr.Warning("Running on CPU will take a much longer time", duration=None)

                yield from progress.tqdm(
                    _pipeline(
                        input_path=inputs[input_3d],
                        is_gs=inputs[input_is_gs],
                        opacity_threshold=inputs[input_opacity_threshold],
                        no_fingers=inputs[input_no_fingers],
                        rest_pose_type=inputs[input_rest_pose],
                        ignore_pose_parts=inputs[input_rest_parts],
                        input_normal=inputs[input_normal],
                        bw_fix=inputs[input_bw_fix],
                        bw_vis_bone=inputs[input_bw_vis_bone],
                        reset_to_rest=inputs[input_reset_to_rest],
                        animation_file=inputs[input_animation_file],
                        retarget=inputs[input_retarget],
                        inplace=inputs[input_inplace],
                        db=inputs[state],
                        # export_temp=True,
                    )
                )
                # gr.Success("Finished successfully!")

            submit_event = submit_btn.click(
                fn=clear_components, inputs=set(outputs), outputs=outputs, show_progress="hidden"
            ).success(
                fn=pipeline, inputs=set(inputs + (state,)), outputs=set(outputs + (state,)), show_progress="minimal"
            )
            animate_event = animate_btn.click(
                fn=clear_components,
                inputs={output_rest_vis, output_anim, output_anim_vis},
                outputs=[output_rest_vis, output_anim, output_anim_vis],
            ).success(
                fn=vis_blender,
                inputs=[
                    input_reset_to_rest,
                    input_no_fingers,
                    input_rest_pose,
                    input_rest_parts,
                    input_animation_file,
                    input_retarget,
                    input_inplace,
                    state,
                ],
                outputs={output_rest_vis, output_anim, output_anim_vis, state},
            )
            animate_event.success(fn=finish, outputs={state})
            stop_btn.click(fn=lambda: [], cancels=[submit_event, animate_event]).success(
                fn=lambda: gr.Warning("Job cancelled") or []
            )
            clear_btn.click(fn=clear_components, inputs=set(outputs), outputs=outputs).success(
                fn=clear, inputs=state, outputs=state
            )

            def select2file(selected: str):
                return None if selected is None else os.path.join(animation_dir, selected)

            select_animation_file.input(
                lambda x: (select2file(x), gr.Checkbox(value=True)),
                inputs=select_animation_file,
                outputs=[input_animation_file, input_reset_to_rest],
                preprocess=False,
            )
            input_animation_file.upload(
                lambda: (gr.Dropdown(value=None), gr.Checkbox(value=True)),
                outputs=[select_animation_file, input_reset_to_rest],
            )
            input_animation_file.clear(
                lambda: gr.Dropdown(value=None), outputs=select_animation_file, show_progress="hidden"
            )
            input_animation_file.change(
                lambda x: (
                    (gr.Checkbox(interactive=True), gr.Checkbox(interactive=True))
                    if x
                    else (gr.Checkbox(interactive=False), gr.Checkbox(interactive=False))
                ),
                inputs=input_animation_file,
                outputs=[input_retarget, input_inplace],
                show_progress="hidden",
            )

            input_retarget.change(
                lambda x: gr.Checkbox(interactive=True) if x else gr.Checkbox(interactive=False),
                inputs=input_retarget,
                outputs=input_inplace,
                show_progress="hidden",
            )

            demo.unload(fn=lambda: not clear(state.value) or None)  # just to make sure fn returns None

    return demo

if __name__ == "__main__":
    # Ensure models are loaded relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Changed working directory to: {script_dir}")

    # Load models once on startup
    init_models()
    
    # Start FastAPI in a background thread
    threading.Thread(target=run_api, daemon=True).start()
    
    # Create the Gradio interface
    demo = init_blocks()
    
    # Launch the Gradio app
    print("Launching Gradio App...")
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
