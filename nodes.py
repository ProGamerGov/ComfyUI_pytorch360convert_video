import math
import os
from typing import Dict, List, Optional, Tuple, Union

# Import ComfyUI's progress utilities
import comfy.utils

import folder_paths
import torch
from pytorch360convert import e2p
from tqdm import tqdm


def _linear_progress(n_frames: int) -> List[float]:
    """
    Generate a linear progression from 0.0 to 1.0 over n_frames.

    Args:
        n_frames (int): Number of frames.

    Returns:
        List[float]: List of normalized progress values.
    """
    return [i / max(1, (n_frames - 1)) for i in range(n_frames)]


def _ease_in_out_progress(n_frames: int) -> List[float]:
    """
    Generate an ease-in-out progression (cosine smoothing) from 0.0 to 1.0.

    Args:
        n_frames (int): Number of frames.

    Returns:
        List[float]: List of normalized progress values.
    """
    # Smooth accelerate then decelerate (cosine easing)
    return [
        0.5 * (1 - math.cos(math.pi * (i / max(1, (n_frames - 1)))))
        for i in range(n_frames)
    ]


def _generate_frames_from_equirect(
    equi_tensors: List[torch.Tensor],
    out_dir: str,
    resolution: Tuple[int, int] = (1080, 1920),
    fps: int = 30,
    duration_per_image: Optional[float] = 4.0,
    total_duration: Optional[float] = None,
    fov_deg: Union[float, Tuple[float, float]] = (70.0, 60.0),
    interpolation_mode: str = "bilinear",
    speed_profile: str = "constant",
    vertical_movement: Optional[Dict] = None,
    device: Optional[torch.device] = None,
    start_frame_index: int = 0,
    save_format: str = "png",
    start_yaw_deg: float = 0.0,
    end_yaw_deg: float = 360.0,
    progress_callback: Optional[callable] = None,
    filename_prefix: str = "frame",  # Added filename_prefix parameter
) -> List[str]:
    """
    Generate video frames by sweeping through one or more
    equirectangular images.

    Args:
        equi_tensors (List[torch.Tensor]): List of equirectangular image tensors.
            Tensors are processed sequentially.

        out_dir (str): Output directory where frames will be saved.

        resolution (tuple of int, optional): Output frame resolution as
            (height, width). Default: (1080, 1920)

        fps (int, optional): Frames per second for timing calculations.
            Default: 30

        duration_per_image (float, optional): Duration in seconds for each
            image sweep. Ignored if total_duration is provided. Default: 4.0

        total_duration (float, optional): Total duration in seconds for all
            images combined. If set, overrides duration_per_image. Default: None

        fov_deg (float or tuple of floats, optional): Field of view in degrees.
            - Single float: same horizontal & vertical FOV.
            - Tuple: (horizontal_fov, vertical_fov).
            Default: (120.0, 100.0)

        interpolation_mode (str, optional): Resampling interpolation.
            Options: "nearest", "bilinear", "bicubic".
            Default: "bilinear"

        speed_profile (str, optional): Progression curve for the sweep.
            Options:
                - "constant": uniform rotation speed.
                - "ease_in_out": slow at start/end, fast in middle.
                - "ease_in": slow at start, speeds up.
                - "ease_out": fast at start, slows down.
            Default: "constant"

        vertical_movement (dict, optional): Parameters for adding pitch movement.
            Dictionary keys:
                - "mode": "none", "during", or "before_after".
                - "amplitude_deg": float, pitch amplitude in degrees.
                - "pattern": "linear" or "sine".
            Set to None or {"mode": "none"} to disable. Default: None

        device (torch.device, optional): Torch device to run on.
            Default: torch.device('cpu')

        start_frame_index (int, optional): Starting frame index for naming
            output files. Useful for concatenating multiple runs.
            Default: 0

        save_format (str, optional): Image format to save frames.
            Options: "png", "jpg", "jpeg", "bmp".
            Default: "png"

        start_yaw_deg (float, optional): Starting yaw angle in degrees.
            Negative = left, positive = right. Default: 0.0

        end_yaw_deg (float, optional): Ending yaw angle in degrees.
            Negative = left, positive = right. Default: 360.0

        progress_callback (callable, optional): Callback function to report progress.
            Should accept (current_step, total_steps, description) arguments.

        filename_prefix (str, optional): Prefix for saved frame filenames.
            Default: "frame"

    Returns:
        List[str]: List of file paths for the saved frames.
    """

    import math

    from PIL import Image

    def _save_tensor_as_image(tensor: torch.Tensor, path: str) -> None:
        """
        Save a CHW float tensor (range [0, 1]) to directory
        """
        if tensor.dim() == 4:  # [B,H,W,C] -> take first
            tensor = tensor[0]
        t = tensor.detach().cpu().clamp(0.0, 1.0) * 255.0
        Image.fromarray(t.to(dtype=torch.uint8).numpy()).save(path)

    os.makedirs(out_dir, exist_ok=True)
    device = device if device is not None else torch.device("cpu")
    saved_paths = []
    n_images = len(equi_tensors)
    if n_images == 0:
        return saved_paths

    # decide frames per image
    if total_duration is not None:
        assert total_duration > 0
        seconds_per_image = total_duration / n_images
    else:
        seconds_per_image = (
            duration_per_image if duration_per_image is not None else 4.0
        )

    frames_per_image = max(1, int(round(seconds_per_image * fps)))

    # Calculate total frames for progress tracking
    total_frames = n_images * frames_per_image

    # Add extra frames for separate pole sweep if enabled
    vm = vertical_movement or {"mode": "none"}
    vm_mode = vm.get("mode", "none")
    if vm_mode == "separate":
        pole_fraction = float(vm.get("pole_fraction", 0.5))
        pole_frames = max(1, int(round(frames_per_image * pole_fraction)))
        total_frames += n_images * pole_frames

    # choose progress function
    if speed_profile == "constant":
        progress_fn = _linear_progress
    elif speed_profile == "ease_in_out":
        progress_fn = _ease_in_out_progress
    else:
        raise ValueError("speed_profile must be 'constant' or 'ease_in_out'")

    frame_idx = start_frame_index
    current_frame = 0
    e2p_jit = e2p  # torch.jit.script(e2p)

    yaw_start, yaw_end = start_yaw_deg, end_yaw_deg

    for img_idx, e_img in enumerate(equi_tensors):
        n = frames_per_image
        prog = progress_fn(n)
        # yaw values from yaw_start -> yaw_end
        yaw_values = [yaw_start + p * (yaw_end - yaw_start) for p in prog]

        # vertical values
        if vm_mode == "during":
            amplitude = float(vm.get("amplitude_deg", 15.0))
            vertical_pattern = vm.get("pattern", "sine")
            if vertical_pattern == "sine":
                v_values = [amplitude * math.sin(2 * math.pi * p) for p in prog]
            else:
                v_values = [amplitude * (2 * p - 1) for p in prog]
        else:
            v_values = [0.0] * n

        # rotation frames
        for i_frame in range(n):
            h_deg = yaw_values[i_frame]
            v_deg = v_values[i_frame]
            pers = e2p_jit(
                e_img,
                fov_deg=fov_deg,
                h_deg=h_deg,
                v_deg=v_deg,
                out_hw=resolution,
                mode=interpolation_mode,
                channels_first=False,
            )
            filename = (
                f"{filename_prefix}_{frame_idx:06d}.{save_format}"  # Use custom prefix
            )
            path = os.path.join(out_dir, filename)
            _save_tensor_as_image(pers, path)
            saved_paths.append(path)
            frame_idx += 1
            current_frame += 1

            # Report progress - focus on frame generation progress
            if progress_callback:
                progress_callback(
                    current_frame,
                    total_frames,
                    f"Generated frame {current_frame}/{total_frames} (Image {img_idx+1}/{n_images})",
                )

        # optional separate pole sweep
        if vm_mode == "separate":
            pole_fraction = float(vm.get("pole_fraction", 0.5))
            pole_frames = max(1, int(round(frames_per_image * pole_fraction)))
            pole_progress = _linear_progress(pole_frames)
            pole_v_values = [(-85.0 + 170.0 * p) for p in pole_progress]
            center_yaw = (yaw_start + yaw_end) / 2.0
            for pole_idx, v_deg in enumerate(pole_v_values):
                pers = e2p(
                    e_img,
                    fov_deg=fov_deg,
                    h_deg=center_yaw,
                    v_deg=v_deg,
                    out_hw=resolution,
                    mode=interpolation_mode,
                    channels_first=False,
                )
                filename = f"{filename_prefix}_{frame_idx:06d}.{save_format}"  # Use custom prefix
                path = os.path.join(out_dir, filename)
                _save_tensor_as_image(pers, path)
                saved_paths.append(path)
                frame_idx += 1
                current_frame += 1

                # Report progress for pole sweep
                if progress_callback:
                    progress_callback(
                        current_frame,
                        total_frames,
                        f"Generated frame {current_frame}/{total_frames} (Pole sweep - Image {img_idx+1}/{n_images})",
                    )

    return saved_paths


class Create360SweepVideoFramesNode:
    def __init__(self) -> None:
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4  # PNG compress level (0-9)

    @classmethod
    def INPUT_TYPES(s) -> Dict:
        return {
            "required": {
                "images": (
                    "IMAGE",
                    {"tooltip": "Input equirectangular images (BHWC tensor)."},
                ),
                "output_dir": (
                    "STRING",
                    {
                        "default": folder_paths.get_output_directory(),
                        "tooltip": "Where to save frames (video_frames subfolder added).",
                    },
                ),
                "filename_prefix": (
                    "STRING",
                    {"default": "sweep360", "tooltip": "Prefix for saved frames"},
                ),
                "width": (
                    "INT",
                    {
                        "default": 1920,
                        "tooltip": "Output frame width (pixels)."
                        + " Should match fov_w:fov_h ratio for best results.",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 1080,
                        "tooltip": "Output frame height (pixels)."
                        + " Should match fov_w:fov_h ratio for best results.",
                    },
                ),
                "fps": ("INT", {"default": 60, "tooltip": "Frames per second."}),
                "duration_per_image": (
                    "FLOAT",
                    {
                        "default": 5.0,
                        "tooltip": "Seconds per 360° sweep (per input image).",
                    },
                ),
                "total_duration": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "tooltip": "Total duration for all images (overrides duration_per_image if >0).",
                    },
                ),
                "fov_h": (
                    "FLOAT",
                    {
                        "default": 120.0,
                        "tooltip": "Horizontal FOV (degrees)."
                        + " Should match width:height ratio for best results.",
                    },
                ),
                "fov_w": (
                    "FLOAT",
                    {
                        "default": 100.0,
                        "tooltip": "Vertical FOV (degrees)."
                        + " Should match width:height ratio for best results.",
                    },
                ),
                "speed_profile": (
                    ["constant", "ease_in_out"],
                    {"default": "constant", "tooltip": "Yaw speed profile."},
                ),
                "interpolation_mode": (
                    ["bilinear", "bicubic", "nearest"],
                    {
                        "default": "bilinear",
                        "tooltip": "Sampling mode for equirectangular to projection.",
                    },
                ),
                "vertical_mode": (
                    ["none", "during", "separate"],
                    {
                        "default": "none",
                        "tooltip": "Vertical movement mode."
                        + " Only used if vertical_mode is not set to None.",
                    },
                ),
                "vertical_amplitude_deg": (
                    "FLOAT",
                    {
                        "default": 15.0,
                        "tooltip": "Amplitude in degrees for vertical motion."
                        + " Only used if vertical_mode is not set to None.",
                    },
                ),
                "vertical_pattern": (
                    ["linear", "sine"],
                    {
                        "default": "sine",
                        "tooltip": "Pattern for vertical motion."
                        + " Only used if vertical_mode is not set to None.",
                    },
                ),
                "vertical_pole_fraction": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "tooltip": "Fraction of frames for separate pole sweep."
                        + " Only used if vertical_mode is not set to None.",
                    },
                ),
                "start_yaw_deg": (
                    "FLOAT",
                    {"default": 0.0, "tooltip": "Starting yaw angle (degrees)."},
                ),
                "end_yaw_deg": (
                    "FLOAT",
                    {"default": 360.0, "tooltip": "Ending yaw angle (degrees)."},
                ),
                "save_format": (
                    ["png", "jpg", "jpeg", "bmp"],
                    {"default": "png", "tooltip": "Image format for saved frames."},
                ),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "generate_frames"
    OUTPUT_NODE = True
    CATEGORY = "pytorch360convert/video"
    DESCRIPTION = "Generate rotational frames from 360° equirectangular images and save them into <output_dir>/video_frames/."

    def generate_frames(
        self,
        images,
        output_dir,
        filename_prefix="sweep360",
        width=1920,
        height=1080,
        fps=30,
        duration_per_image=5.0,
        total_duration=0.0,
        fov_h=120.0,
        fov_w=100.0,
        speed_profile="constant",
        interpolation_mode="bilinear",
        vertical_mode="none",
        vertical_amplitude_deg=15.0,
        vertical_pattern="sine",
        vertical_pole_fraction=0.5,
        start_yaw_deg=0.0,
        end_yaw_deg=360.0,
        save_format="png",
        prompt=None,
        extra_pnginfo=None,
    ):
        """
        Wrapper that converts Comfy UI inputs to the external generate_frames_from_equirect function.
        """

        # Prepare output directory
        video_dir_parent = output_dir or self.output_dir
        video_frames_dir = os.path.join(video_dir_parent, "video_frames")
        os.makedirs(video_frames_dir, exist_ok=True)

        # Save images temporarily to disk if they are tensors
        equi_tensors = []
        for img in images:
            equi_tensors.append(img)

        # Construct vertical movement dictionary
        vertical_movement = {
            "mode": vertical_mode,
            "amplitude_deg": vertical_amplitude_deg,
            "pattern": vertical_pattern,
            "pole_fraction": vertical_pole_fraction,
        }

        # Calculate total frames for progress bar initialization
        n_images = len(equi_tensors)
        if total_duration > 0:
            seconds_per_image = total_duration / n_images
        else:
            seconds_per_image = duration_per_image

        frames_per_image = max(1, int(round(seconds_per_image * fps)))
        total_frames = n_images * frames_per_image

        # Add extra frames for separate pole sweep if enabled
        if vertical_mode == "separate":
            pole_frames = max(1, int(round(frames_per_image * vertical_pole_fraction)))
            total_frames += n_images * pole_frames

        # Check if progress bar is enabled
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        pbar = None

        if not disable_pbar:
            # Initialize ComfyUI progress bar
            pbar = comfy.utils.ProgressBar(total_frames)

        def progress_callback(current, total, description):
            """Callback function to update ComfyUI progress bar"""
            if pbar is not None:
                pbar.update_absolute(current, total, preview=None)

        # Call the function with progress callback and filename_prefix
        saved_paths = _generate_frames_from_equirect(
            equi_tensors=equi_tensors,
            out_dir=video_frames_dir,
            resolution=(height, width),
            fps=fps,
            duration_per_image=duration_per_image,
            total_duration=total_duration if total_duration > 0 else None,
            fov_deg=(fov_h, fov_w),
            interpolation_mode=interpolation_mode,
            speed_profile=speed_profile,
            vertical_movement=vertical_movement,
            start_frame_index=0,
            save_format=save_format,
            start_yaw_deg=start_yaw_deg,
            end_yaw_deg=end_yaw_deg,
            progress_callback=progress_callback,
            filename_prefix=filename_prefix,  # Pass the filename_prefix parameter
        )

        saved_paths = saved_paths[0:8] if len(saved_paths) > 8 else saved_paths

        # Return results for Comfy UI
        results = [
            {
                "filename": os.path.basename(p),
                "subfolder": "video_frames",
                "type": self.type,
            }
            for p in saved_paths
        ]
        return {"ui": {"images": results}}


###############################


def _generate_frames_from_equirect_tensors(
    equi_tensors: List[torch.Tensor],
    resolution: Tuple[int, int] = (1080, 1920),
    fps: int = 30,
    duration_per_image: Optional[float] = 4.0,
    total_duration: Optional[float] = None,
    fov_deg: Union[float, Tuple[float, float]] = (70.0, 60.0),
    interpolation_mode: str = "bilinear",
    speed_profile: str = "constant",
    vertical_movement: Optional[Dict] = None,
    device: Optional[torch.device] = None,
    start_yaw_deg: float = 0.0,
    end_yaw_deg: float = 360.0,
    progress_callback: Optional[callable] = None,
) -> torch.Tensor:
    """
    Generate video frames by sweeping through one or more
    equirectangular images.

    Args:
        equi_tensors (List[torch.Tensor]): List of equirectangular image tensors.
            Tensors are processed sequentially.

        resolution (tuple of int, optional): Output frame resolution as
            (height, width). Default: (1080, 1920)

        fps (int, optional): Frames per second for timing calculations.
            Default: 30

        duration_per_image (float, optional): Duration in seconds for each
            image sweep. Ignored if total_duration is provided. Default: 4.0

        total_duration (float, optional): Total duration in seconds for all
            images combined. If set, overrides duration_per_image. Default: None

        fov_deg (float or tuple of floats, optional): Field of view in degrees.
            - Single float: same horizontal & vertical FOV.
            - Tuple: (horizontal_fov, vertical_fov).
            Default: (120.0, 100.0)

        interpolation_mode (str, optional): Resampling interpolation.
            Options: "nearest", "bilinear", "bicubic".
            Default: "bilinear"

        speed_profile (str, optional): Progression curve for the sweep.
            Options:
                - "constant": uniform rotation speed.
                - "ease_in_out": slow at start/end, fast in middle.
                - "ease_in": slow at start, speeds up.
                - "ease_out": fast at start, slows down.
            Default: "constant"

        vertical_movement (dict, optional): Parameters for adding pitch movement.
            Dictionary keys:
                - "mode": "none", "during", or "before_after".
                - "amplitude_deg": float, pitch amplitude in degrees.
                - "pattern": "linear" or "sine".
            Set to None or {"mode": "none"} to disable. Default: None

        device (torch.device, optional): Torch device to run on.
            Default: torch.device('cpu')

        save_format (str, optional): Image format to save frames.
            Options: "png", "jpg", "jpeg", "bmp".
            Default: "png"

        start_yaw_deg (float, optional): Starting yaw angle in degrees.
            Negative = left, positive = right. Default: 0.0

        end_yaw_deg (float, optional): Ending yaw angle in degrees.
            Negative = left, positive = right. Default: 360.0

        progress_callback (callable, optional): Callback function to report progress.
            Should accept (current_step, total_steps, description) arguments.

    Returns:
        torch.Tensor: Stack of frames
    """

    device = device if device is not None else torch.device("cpu")
    frame_tensors = []
    n_images = len(equi_tensors)

    # decide frames per image
    if total_duration is not None:
        assert total_duration > 0
        seconds_per_image = total_duration / n_images
    else:
        seconds_per_image = (
            duration_per_image if duration_per_image is not None else 4.0
        )

    frames_per_image = max(1, int(round(seconds_per_image * fps)))

    # Calculate total frames for progress tracking
    total_frames = n_images * frames_per_image

    # Add extra frames for separate pole sweep if enabled
    vm = vertical_movement or {"mode": "none"}
    vm_mode = vm.get("mode", "none")
    if vm_mode == "separate":
        pole_fraction = float(vm.get("pole_fraction", 0.5))
        pole_frames = max(1, int(round(frames_per_image * pole_fraction)))
        total_frames += n_images * pole_frames

    # choose progress function
    if speed_profile == "constant":
        progress_fn = _linear_progress
    elif speed_profile == "ease_in_out":
        progress_fn = _ease_in_out_progress
    else:
        raise ValueError("speed_profile must be 'constant' or 'ease_in_out'")

    current_frame = 0
    e2p_jit = e2p  # torch.jit.script(e2p)

    yaw_start, yaw_end = start_yaw_deg, end_yaw_deg

    for img_idx, e_img in enumerate(equi_tensors):
        n = frames_per_image
        prog = progress_fn(n)
        # yaw values from yaw_start -> yaw_end
        yaw_values = [yaw_start + p * (yaw_end - yaw_start) for p in prog]

        # vertical values
        if vm_mode == "during":
            amplitude = float(vm.get("amplitude_deg", 15.0))
            vertical_pattern = vm.get("pattern", "sine")
            if vertical_pattern == "sine":
                v_values = [amplitude * math.sin(2 * math.pi * p) for p in prog]
            else:
                v_values = [amplitude * (2 * p - 1) for p in prog]
        else:
            v_values = [0.0] * n

        # rotation frames
        for i_frame in range(n):
            h_deg = yaw_values[i_frame]
            v_deg = v_values[i_frame]
            pers = e2p_jit(
                e_img,
                fov_deg=fov_deg,
                h_deg=h_deg,
                v_deg=v_deg,
                out_hw=resolution,
                mode=interpolation_mode,
                channels_first=False,
            )
            frame_tensors.append(pers)
            current_frame += 1

            # Report progress - focus on frame generation progress
            if progress_callback:
                progress_callback(
                    current_frame,
                    total_frames,
                    f"Generated frame {current_frame}/{total_frames} (Image {img_idx+1}/{n_images})",
                )

        # optional separate pole sweep
        if vm_mode == "separate":
            pole_fraction = float(vm.get("pole_fraction", 0.5))
            pole_frames = max(1, int(round(frames_per_image * pole_fraction)))
            pole_progress = _linear_progress(pole_frames)
            pole_v_values = [(-85.0 + 170.0 * p) for p in pole_progress]
            center_yaw = (yaw_start + yaw_end) / 2.0
            for pole_idx, v_deg in enumerate(pole_v_values):
                pers = e2p(
                    e_img,
                    fov_deg=fov_deg,
                    h_deg=center_yaw,
                    v_deg=v_deg,
                    out_hw=resolution,
                    mode=interpolation_mode,
                    channels_first=False,
                )
                pers = e2p(
                    e_img,
                    fov_deg=fov_deg,
                    h_deg=center_yaw,
                    v_deg=v_deg,
                    out_hw=resolution,
                    mode=interpolation_mode,
                    channels_first=False,
                )
                frame_tensors.append(pers)
                current_frame += 1

                # Report progress for pole sweep
                if progress_callback:
                    progress_callback(
                        current_frame,
                        total_frames,
                        f"Generated frame {current_frame}/{total_frames} (Pole sweep - Image {img_idx+1}/{n_images})",
                    )

    # Stack all frames into a single tensor (N, H, W, C)
    if frame_tensors:
        video_tensor = torch.stack(frame_tensors, dim=0)
    else:
        video_tensor = torch.zeros(
            (0, resolution[0], resolution[1], 3), dtype=equi_tensors.dtype, device=device
        )

    return video_tensor


class Create360SweepVideoFramesTensorNode:
    @classmethod
    def INPUT_TYPES(s) -> Dict:
        return {
            "required": {
                "image": (
                    "IMAGE",
                    {"tooltip": "Input equirectangular images (BHWC tensor)."},
                ),
                "width": (
                    "INT",
                    {
                        "default": 1920,
                        "tooltip": "Output frame width (pixels)."
                        + " Should match fov_w:fov_h ratio for best results.",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 1080,
                        "tooltip": "Output frame height (pixels)."
                        + " Should match fov_w:fov_h ratio for best results.",
                    },
                ),
                "fps": ("INT", {"default": 60, "tooltip": "Frames per second."}),
                "duration_per_image": (
                    "FLOAT",
                    {
                        "default": 5.0,
                        "tooltip": "Seconds per 360° sweep (per input image).",
                    },
                ),
                "total_duration": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "tooltip": "Total duration for all images (overrides duration_per_image if >0).",
                    },
                ),
                "fov_h": (
                    "FLOAT",
                    {
                        "default": 120.0,
                        "tooltip": "Horizontal FOV (degrees)."
                        + " Should match width:height ratio for best results.",
                    },
                ),
                "fov_w": (
                    "FLOAT",
                    {
                        "default": 100.0,
                        "tooltip": "Vertical FOV (degrees)."
                        + " Should match width:height ratio for best results.",
                    },
                ),
                "speed_profile": (
                    ["constant", "ease_in_out"],
                    {"default": "constant", "tooltip": "Yaw speed profile."},
                ),
                "interpolation_mode": (
                    ["bilinear", "bicubic", "nearest"],
                    {
                        "default": "bilinear",
                        "tooltip": "Sampling mode for equirectangular to projection.",
                    },
                ),
                "vertical_mode": (
                    ["none", "during", "separate"],
                    {
                        "default": "none",
                        "tooltip": "Vertical movement mode."
                        + " Only used if vertical_mode is not set to None.",
                    },
                ),
                "vertical_amplitude_deg": (
                    "FLOAT",
                    {
                        "default": 15.0,
                        "tooltip": "Amplitude in degrees for vertical motion."
                        + " Only used if vertical_mode is not set to None.",
                    },
                ),
                "vertical_pattern": (
                    ["linear", "sine"],
                    {
                        "default": "sine",
                        "tooltip": "Pattern for vertical motion."
                        + " Only used if vertical_mode is not set to None.",
                    },
                ),
                "vertical_pole_fraction": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "tooltip": "Fraction of frames for separate pole sweep."
                        + " Only used if vertical_mode is not set to None.",
                    },
                ),
                "start_yaw_deg": (
                    "FLOAT",
                    {"default": 0.0, "tooltip": "Starting yaw angle (degrees)."},
                ),
                "end_yaw_deg": (
                    "FLOAT",
                    {"default": 360.0, "tooltip": "Ending yaw angle (degrees)."},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Video Frames",)

    FUNCTION = "generate_frames"
    OUTPUT_NODE = False
    CATEGORY = "pytorch360convert/video"
    DESCRIPTION = (
        "Generate rotational frames from 360° equirectangular images and"
        + " return them as a stack of tensors"
    )

    def generate_frames(
        self,
        image,
        width=1920,
        height=1080,
        fps=30,
        duration_per_image=5.0,
        total_duration=0.0,
        fov_h=120.0,
        fov_w=100.0,
        speed_profile="constant",
        interpolation_mode="bilinear",
        vertical_mode="none",
        vertical_amplitude_deg=15.0,
        vertical_pattern="sine",
        vertical_pole_fraction=0.5,
        start_yaw_deg=0.0,
        end_yaw_deg=360.0,
    ):
        """
        Wrapper that converts Comfy UI inputs to the external
        generate_frames_from_equirect_tensors function.
        """

        assert image.shape[0] == 1 and image.dim() == 4 or image.dim() == 3

        # Construct vertical movement dictionary
        vertical_movement = {
            "mode": vertical_mode,
            "amplitude_deg": vertical_amplitude_deg,
            "pattern": vertical_pattern,
            "pole_fraction": vertical_pole_fraction,
        }

        # Calculate total frames for progress bar initialization
        n_images = len(image)
        if total_duration > 0:
            seconds_per_image = total_duration / n_images
        else:
            seconds_per_image = duration_per_image

        frames_per_image = max(1, int(round(seconds_per_image * fps)))
        total_frames = n_images * frames_per_image

        # Add extra frames for separate pole sweep if enabled
        if vertical_mode == "separate":
            pole_frames = max(1, int(round(frames_per_image * vertical_pole_fraction)))
            total_frames += n_images * pole_frames

        # Check if progress bar is enabled
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        pbar = None

        if not disable_pbar:
            # Initialize ComfyUI progress bar
            pbar = comfy.utils.ProgressBar(total_frames)

        def progress_callback(current, total, description):
            """Callback function to update ComfyUI progress bar"""
            if pbar is not None:
                pbar.update_absolute(current, total, preview=None)

        # Call the function with progress callback and filename_prefix
        video_frames = _generate_frames_from_equirect_tensors(
            equi_tensors=image,
            resolution=(height, width),
            fps=fps,
            duration_per_image=duration_per_image,
            total_duration=total_duration if total_duration > 0 else None,
            fov_deg=(fov_h, fov_w),
            interpolation_mode=interpolation_mode,
            speed_profile=speed_profile,
            vertical_movement=vertical_movement,
            start_yaw_deg=start_yaw_deg,
            end_yaw_deg=end_yaw_deg,
            progress_callback=progress_callback,
        )

        return (video_frames,)
