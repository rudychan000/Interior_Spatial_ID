import json
import numpy as np
from ultralytics import YOLO
import pyk4a
from pyk4a import FPS, CalibrationType, Config, PyK4A
from pyproj import Transformer
import cv2
from spatialID import generate_ids_for_objects

import asyncio
import websockets
import time

# ---- constants ---------------------------------------------------
MODEL_PATH = "yolo_model/yolo11m.pt"   
T_cam2ECEF = np.load("trans/T_cam2ECEF.npy") 
_T_ECEF2GEO = Transformer.from_crs("EPSG:4978", "EPSG:4979", always_xy=True)
# ------------------------------------------------------------------

def start_camera():
    """Initialize the Azure Kinect camera."""
    k4a = PyK4A(Config(color_resolution=pyk4a.ColorResolution.RES_1536P,
                       depth_mode=pyk4a.DepthMode.NFOV_2X2BINNED,
                       camera_fps=FPS.FPS_30))
    k4a.start()
    print("Azure Kinect started.")
    return k4a

def load_model():
    """Load the YOLO model."""
    model = YOLO(MODEL_PATH)
    model.to('cuda')  # move model to GPU
    print(f"YOLO v11 model {MODEL_PATH} loaded on {model.device}.")
    return model

def xyz_to_ecef(xyz):
    """
    Convert local camera coordinates (x, y, z) in metres to ECEF coordinates.
    """
    # Apply the camera to ECEF transformation
    xyz_h = np.array(xyz + (1.0,))  # Convert to homogeneous coordinates
    ecef_h = T_cam2ECEF @ xyz_h.T  # Apply transformation
    return ecef_h[:3].T  # Return only the first three rows (x, y, z)


def detect_objects(capture, model, k4a):
    """
    Detect objects in the given capture using YOLO.
    """
    rgb = cv2.cvtColor(capture.color, cv2.COLOR_BGRA2RGB)
    #current_time = time.time()
    depth = capture.transformed_depth  # depth aligned to color image
    H, W = rgb.shape[:2]

    #print(f"Point cloud extraction time: {time.time() - current_time:.3f} seconds")
    # YOLO detection on RGB
    preds = model(rgb, verbose=False)[0]
    objects = []

    for xyxy, cls_id, conf in zip(preds.boxes.xyxy,
                                preds.boxes.cls,
                                preds.boxes.conf):
        x1, y1, x2, y2 = map(int, xyxy.cpu().numpy())
        u_c, v_c = (x1 + x2) // 2, (y1 + y2) // 2
        u_c = np.clip(u_c, 0, W - 1)
        v_c = np.clip(v_c, 0, H - 1)
        # Extract a small window around the center pixel to find depth
        # This is a 5x5 window centered at (u_c, v_c)
        win = depth[max(v_c-2, 0):v_c+3, max(u_c-2, 0):u_c+3]
        mask = win > 0
        if not np.any(mask):
            continue

        idx = np.nanargmin(np.where(mask, win, np.nan))
        r, c = divmod(idx, win.shape[1])
        u_d = np.clip(u_c - 2 + c, 0, W - 1)
        v_d = np.clip(v_c - 2 + r, 0, H - 1)
        z_mm = depth[v_d, u_d]

        # Convert 2D to 3D using calibration
        point3d = k4a.calibration.convert_2d_to_3d(
            (u_d, v_d), z_mm / 1000.0, CalibrationType.COLOR)  # in meters
        X, Y, Z = point3d

        # Convert to ECEF coordinates
        Xt, Yt, Zt = xyz_to_ecef((X, Y, Z))
        # Convert to latitude, longitude, altitude
        lon, lat, alt = _T_ECEF2GEO.transform(Xt, Yt, Zt)

        objects.append({
            #"bbox": (x1, y1, x2, y2),
            "name": model.names[int(cls_id)],
            "confidence": f"{conf.item():.2f}",
            #"camera_space": (X, Y, Z),
            "ecef": (Xt, Yt, Zt),
            "llh": (lon, lat, alt)
        })
    return objects

async def send_data_to_websocket(data):
    """
    Send data to a WebSocket server.
    """
    uri = "wss://nodered.tlab.cloud/spatial_id"
    async with websockets.connect(uri) as websocket:
        await websocket.send(data)


async def main():
    k4a = start_camera()
    model = load_model()
    uri = "wss://nodered.tlab.cloud/spatial_id"
    async with websockets.connect(uri) as websocket:
        try:
            frame_count = 0
            start_time = time.time()
            while True:
                capture = k4a.get_capture()
                if capture.color is None or capture.depth is None:
                    continue
                
                #current_time = time.time()

                objects = detect_objects(capture, model,k4a)
                if not objects:
                    continue
                
                #print(f"Detection time: {time.time() - current_time:.3f} seconds")

                # for _ in range(10):
                #     objects.append(objects[0])

                #current_time = time.time()


                objects_with_ids = generate_ids_for_objects(objects)

                #print((f"Id computation time: {time.time() - current_time:.3f} seconds"))

                json_data = json.dumps(objects_with_ids)

                # Send without blocking
                await websocket.send(json_data)

                # FPS monitor
                frame_count += 1
                elapsed = time.time() - start_time
                if elapsed >= 1.0:
                    print(f"FPS: {frame_count / elapsed:.2f}")
                    frame_count = 0
                    start_time = time.time()
        finally:
            k4a.stop()

if __name__ == "__main__":
    asyncio.run(main())