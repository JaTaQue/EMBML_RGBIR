"""
WBS 3.2: RGBT-Tiny Camera Stream Simulator
Reads paired RGB (00) and Thermal (01) frame sequences from the sequence folder,
resizes them, and stacks them into a 4-channel tensor [H, W, 4] at target 15 FPS.
"""

import os
import glob
import time
import cv2
import numpy as np

class RGBTinyStreamSimulator:
    def __init__(self, sequence_dir: str, target_fps: float = 15.0):
        """
        Args:
            sequence_dir (str): Path to sequence directory (e.g. '.../images/DJI_0022_1')
            target_fps (float): Desired playback frame rate (default: 15.0 FPS)
        """
        self.target_fps = target_fps
        self.frame_delay = 1.0 / target_fps

        # Subfolders for RGB (00) and Thermal/IR (01)
        self.rgb_dir = os.path.join(sequence_dir, "00")
        self.ir_dir = os.path.join(sequence_dir, "01")

        if not os.path.exists(self.rgb_dir):
            raise FileNotFoundError(f"RGB folder not found: {self.rgb_dir}")
        if not os.path.exists(self.ir_dir):
            raise FileNotFoundError(f"IR folder not found: {self.ir_dir}")

        # Gather and sort all frame paths
        extensions = ('*.jpg', '*.png', '*.bmp', '*.jpeg')
        self.rgb_files = []
        self.ir_files = []

        for ext in extensions:
            self.rgb_files.extend(glob.glob(os.path.join(self.rgb_dir, ext)))
            self.ir_files.extend(glob.glob(os.path.join(self.ir_dir, ext)))

        self.rgb_files.sort()
        self.ir_files.sort()

        if len(self.rgb_files) == 0:
            raise FileNotFoundError(f"No RGB images found in {self.rgb_dir}")
        if len(self.ir_files) == 0:
            raise FileNotFoundError(f"No IR images found in {self.ir_dir}")

        self.num_frames = min(len(self.rgb_files), len(self.ir_files))
        self.current_idx = 0

        print(f"[Stream Simulator] Loaded sequence '{os.path.basename(sequence_dir)}': "
              f"{self.num_frames} paired frames at {target_fps} FPS.")

    def read_fused_frame(self, target_size=(640, 640)):
        """
        Reads next frame pair (00 and 01), converts IR to 1-channel,
        and stacks them into a 4-channel array [H, W, 4] (RGB + IR).

        Returns:
            fused_4ch (np.ndarray): Shape (640, 640, 4) tensor
            raw_rgb (np.ndarray): Original RGB frame for visualization
            success (bool): True if frame read succeeded
        """
        if self.current_idx >= self.num_frames:
            return None, None, False

        rgb_path = self.rgb_files[self.current_idx]
        ir_path = self.ir_files[self.current_idx]
        self.current_idx += 1

        frame_rgb = cv2.imread(rgb_path)
        frame_ir = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)

        if frame_rgb is None or frame_ir is None:
            print(f"[Error] Could not read frame pair: {rgb_path} / {ir_path}")
            return None, None, False

        # 1. Resize RGB image to network input size
        rgb_resized = cv2.resize(frame_rgb, target_size)

        # 2. Resize Thermal frame to match RGB input size
        ir_resized = cv2.resize(frame_ir, target_size)

        # 3. Expand IR channel to (H, W, 1) and stack -> (H, W, 4)
        ir_channel = np.expand_dims(ir_resized, axis=-1)
        fused_4ch = np.concatenate([rgb_resized, ir_channel], axis=-1)

        return fused_4ch, frame_rgb, True

    def reset(self):
        """Reset sequence playback to start of folder."""
        self.current_idx = 0