import os
import numpy as np
import trimesh
import pyrender

# from OpenGL import EGL
# print("EGL version:", EGL.eglQueryString(EGL.EGL_DEFAULT_DISPLAY, EGL.EGL_VERSION))

import open3d as o3d
import numpy as np

# --------------------------------------------------
# 1. Load your PLY point cloud
# --------------------------------------------------
pcd = o3d.io.read_point_cloud("420673_laser_scan_cropped.ply")   # must have xyz + optional rgb
print(pcd)

# --------------------------------------------------
# 2. Setup the headless renderer
# --------------------------------------------------
w, h = 1920, 1080
renderer = o3d.visualization.rendering.OffscreenRenderer(w, h)

# Create a scene and add the point cloud
mat = o3d.visualization.rendering.MaterialRecord()
mat.shader = "defaultUnlit"    # simple color shader
renderer.scene.add_geometry("cloud", pcd, mat)

# --------------------------------------------------
# 3. Setup camera
# --------------------------------------------------
# Example: 4x4 extrinsic matrix (world → camera)
# Replace with your homogeneous matrix
extrinsic = np.eye(4)

# Example intrinsics (fx, fy, cx, cy)
fx, fy = 1000, 1000
cx, cy = w/2, h/2
intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)

# Apply camera
renderer.setup_camera(intrinsic, extrinsic)

# --------------------------------------------------
# 4. Render and save
# --------------------------------------------------
img = renderer.render_to_image()
o3d.io.write_image("render.png", img)

print("Saved render.png")
