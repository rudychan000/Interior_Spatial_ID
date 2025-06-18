"""
camera_to_world.py
------------------
Compute and save the 4x4 homogeneous transform that takes Azure-Kinect
camera-space coordinates straight to Earth-Centred, Earth-Fixed (ECEF).
"""

import numpy as np
import cv2
from pyproj import Transformer
from pyapriltags import Detector
from pyk4a import FPS, PyK4A, Config, ColorResolution, DepthMode, CalibrationType

# ---- constants ---------------------------------------------------
TAG_SIZE_M   = 0.116           # physical edge length of the AprilTag [m]
TAG_LAT_DEG  = 34.675675   # latitude  (WGS-84) [°]
TAG_LON_DEG  = 135.503916      # longitude (WGS-84) [°]
TAG_ALT_M    = 14.0            # ellipsoidal height [m]
TAG_YAW_DEG  = 0.0            # yaw of tag +X(Right to the tag) measured anticlockwise from +E axis

# ---- camera acquisition -------------------------------------------------------
K4A_CONFIG = Config(                       
    color_resolution=ColorResolution.RES_1536P,
    depth_mode=DepthMode.NFOV_2X2BINNED,
    camera_fps=FPS.FPS_30
)
# ---- transformer from geodetic to ECEF -----------------------------------
_T_GEO2ECEF = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)


def grab_frame_and_detect():
    """Acquire one RGB frame, detect the AprilTag, return detection + intrinsics."""
    k4a = PyK4A(K4A_CONFIG)
    k4a.start()
    try:
        cap = k4a.get_capture()
        rgb = cap.color
        if rgb is None:
            raise RuntimeError("No RGB frame captured.")

        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        Kmat = k4a.calibration.get_camera_matrix(CalibrationType.COLOR)  # 3×3
        fx, fy, cx, cy = Kmat[0, 0], Kmat[1, 1], Kmat[0, 2], Kmat[1, 2]
    finally:
        k4a.stop()

    dets = Detector(families="tag36h11").detect(
        gray,
        estimate_tag_pose=True,
        camera_params=(fx, fy, cx, cy),
        tag_size=TAG_SIZE_M,
    )
    if not dets:
        raise RuntimeError("AprilTag not found in the frame.")
    return dets[0]

def build_T_cam2tag(det):
    T_tag2cam = np.eye(4, dtype=np.float64)
    T_tag2cam[:3, :3] = det.pose_R
    T_tag2cam[:3,  3] = det.pose_t.flatten()
    # Invert ⇒ Camera→Tag
    T_cam2tag = np.linalg.inv(T_tag2cam)
    return T_cam2tag

def build_T_tag2ENU():
    yaw = np.radians(TAG_YAW_DEG)
    c, s = np.cos(yaw), np.sin(yaw)

    T_tag2enu = np.eye(4, dtype=np.float64)
    T_tag2enu[:3, :3] = np.column_stack((
        np.array([ c,  s, 0.0]),     # tag +X (right)
        np.array([ 0.0, 0.0, -1.0]),  # tag +Y (down)
        np.array([-s,  c, 0.0])      # tag +Z (into wall)
    ))
    T_tag2enu[:3, 3] = np.zeros(3)  # origin at tag
    return T_tag2enu

def build_T_cam2ENU(T_cam2tag, T_tag2ENU):
    return T_tag2ENU @ T_cam2tag

def build_T_ENU2ECEF():
    lat = np.radians(TAG_LAT_DEG)
    lon = np.radians(TAG_LON_DEG)
    x0, y0, z0 = _T_GEO2ECEF.transform(TAG_LON_DEG, TAG_LAT_DEG, TAG_ALT_M)
    # Define ENU to ECEF rotation matrix
    R = np.array([
        [-np.sin(lon), -np.sin(lat)*np.cos(lon),  np.cos(lat)*np.cos(lon)],
        [ np.cos(lon), -np.sin(lat)*np.sin(lon),  np.cos(lat)*np.sin(lon)],
        [          0,              np.cos(lat),              np.sin(lat)]
    ])
    T_ENU2ECEF = np.eye(4, dtype=np.float64)
    T_ENU2ECEF[:3, :3] = R
    T_ENU2ECEF[:3, 3] = np.array([x0, y0, z0])  # origin in ECEF
    return T_ENU2ECEF

def build_T_cam2ECEF(T_cam2ENU, T_ENU2ECEF):
    # 
    return T_ENU2ECEF @ T_cam2ENU



# --------------------------------------------------------------------  main workflow
def main():
    print("Capturing frame and detecting AprilTag …")
    det = grab_frame_and_detect()

    T_cam2tag = build_T_cam2tag(det)
    T_tag2ENU = build_T_tag2ENU()
    T_cam2ENU = build_T_cam2ENU(T_cam2tag, T_tag2ENU)
    T_ENU2ECEF = build_T_ENU2ECEF()
    T_cam2ECEF = build_T_cam2ECEF(T_cam2ENU, T_ENU2ECEF)

    transforms = {
        "T_cam2tag":  T_cam2tag,
        "T_tag2ENU":  T_tag2ENU,
        "T_cam2ENU":  T_cam2ENU,
        "T_ENU2ECEF": T_ENU2ECEF,
        "T_cam2ECEF": T_cam2ECEF,
    }
    for name, T in transforms.items():
        np.save(f"trans/{name}.npy", T)

    # --- test ------------------------------------------------------
    cam_origin = np.array([0.0, 0.0, 0.0, 1.0])      # camera itself
    world_origin = T_cam2ECEF @ cam_origin
    world_origin = world_origin[:3]
    _T_ECEF2GEO = Transformer.from_crs("EPSG:4978", "EPSG:4979", always_xy=True)
    lon, lat, alt = _T_ECEF2GEO.transform(world_origin[0], world_origin[1], world_origin[2])
    print(f"\nCamera origin: lon={lon:.6f}°  lat={lat:.6f}°  alt={alt:.3f} m")

# --------------------------------------------------------------------  run as script
if __name__ == "__main__":
    main()
