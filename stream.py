import cv2
import numpy as np
from ultralytics import YOLO
import pyk4a
from pyk4a import Config, PyK4A
from pyproj import Transformer

# ── configuration ───────────────────────────────────────────────────────────
MODEL_PATH = "yolo_model/yolo11m.pt"   
T_cam2ECEF = np.load("trans/T_cam2ECEF.npy") 
_T_ECEF2GEO = Transformer.from_crs("EPSG:4978", "EPSG:4979", always_xy=True)

def start_camera():
    """Initialize the Azure Kinect camera."""
    k4a = PyK4A(Config(color_resolution=pyk4a.ColorResolution.RES_1536P,
                       depth_mode=pyk4a.DepthMode.NFOV_2X2BINNED))
    k4a.start()
    print("Azure Kinect started.")
    return k4a

def load_model():
    """Load the YOLO model."""
    model = YOLO(MODEL_PATH)
    print(f"YOLO v11 model {MODEL_PATH} loaded.")
    return model

def xyz_to_ecef(xyz):
    """
    Convert local camera coordinates (x, y, z) in metres to ECEF coordinates.
    """
    # Apply the camera to ECEF transformation
    xyz_h = np.array(xyz + (1.0,))  # Convert to homogeneous coordinates
    ecef_h = T_cam2ECEF @ xyz_h.T  # Apply transformation
    return ecef_h[:3].T  # Return only the first three rows (x, y, z)

def main():
    k4a = start_camera()
    model = load_model()

    cv2.namedWindow("K4a", cv2.WINDOW_NORMAL)

    try:
        while True:
            capture = k4a.get_capture()

            # Check if the capture contains valid color and depth images
            if capture.color is None or capture.depth is None:
                continue
            rgb = cv2.cvtColor(capture.color, cv2.COLOR_BGRA2RGB)
            points = capture.transformed_depth_point_cloud
            xyz_m = points.astype(np.float32) / 1000.0
            # YOLO detection on RGB
            preds = model(rgb, verbose=False)[0]
            H, W = rgb.shape[:2]
            rgb_overlay = rgb.copy()
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
                win = xyz_m[max(v_c-2,0):v_c+3, max(u_c-2,0):u_c+3]
                z_win = win[..., 2]
                mask = z_win > 0
                if not np.any(mask):
                            # --- keep the box but flag no‑depth ---
                    cv2.rectangle(rgb_overlay, (x1,y1), (x2,y2), (0,0,255), 2)  
                    cv2.putText(rgb_overlay, "No depth", (x1, y2+15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                    continue

                idx = np.nanargmin(np.where(mask, z_win, np.nan))
                r, c = divmod(idx, z_win.shape[1])
                X, Y, Z = win[r, c]

                # Convert to ECEF coordinates
                ecef_coords = xyz_to_ecef((X, Y, Z))
                Xt, Yt, Zt = ecef_coords
                # Convert to latitude, longitude, altitude
                lon, lat, alt = _T_ECEF2GEO.transform(Xt, Yt, Zt)

                # 2‑D overlay
                cv2.rectangle(rgb_overlay, (x1, y1), (x2, y2), (0,255,0), 2)
                label  = f"{model.names[int(cls_id)]} {conf:.2f}"
                camera_space  = f"CameraSpace: ({X:.2f},{Y:.2f},{Z:.2f})m"
                ecef_space = f"ECEF: ({Xt:.2f}, {Yt:.2f}, {Zt:.2f}) m"
                llh = f"LLH: ({lon:.6f}, {lat:.6f}, {alt:.2f})"
                cv2.putText(rgb_overlay, label, (x1, y1-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                cv2.putText(rgb_overlay, camera_space, (x1, y1+40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,128,255), 2)
                cv2.putText(rgb_overlay, ecef_space, (x1, y1+80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
                cv2.putText(rgb_overlay, llh, (x1, y1+120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,64), 2)
                objects.append({
                    #"bbox": (x1, y1, x2, y2),
                    "name": model.names[int(cls_id)],
                    "confidence": conf,
                    #"camera_space": (X, Y, Z),
                    "ecef": (Xt, Yt, Zt),
                    "llh": (lon, lat, alt)
                })
            cv2.imshow("K4a", cv2.cvtColor(rgb_overlay, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        k4a.stop()
        cv2.destroyAllWindows()
        print("Camera stopped and windows closed.")

if __name__ == "__main__":
    main()