import datetime
import math
import re
from unittest import result
from pyproj import Transformer
import time

# ---- constants ---------------------------------------------------
TARGET_ZOOM_LEVELS = [25,26,27,28]

SIZE = {
    "person": (0.5, 1.7, 0.5),   # (width, height, depth) in meters
    "chair": (0.5, 0.9, 0.5)
}

FLOOR_HEIGHT = 0.0 # To aligm objects to the floor, set to 0.0 for ground level

# ------------------------------------------------------------------

def generate_ids_for_objects(objects):
    """
    Generate target zoom level spatial IDs for a list of objects.
    
    Args:
        objects (list[dict]): List of dicts containing (name, confidence, ecef, llh).
        
    Returns:
        list: List of dict with spatial IDs, name, confidence.
    """
    
    objects_with_ids = []
    for obj in objects:
        name = obj['name']
        # ecef = obj['ecef']
        llh = obj['llh']
        confidence = obj['confidence']

        if name not in SIZE:
            continue  # Skip objects not in predefined sizes

        size = SIZE[name]
        bbox_size_m = (size[0], size[2], size[1])  # (width, depth, height)

        # Generate spatial IDs
        #ids = biuld_ids(llh, bbox_size_m)

        # Use corner IDs for better performance
        ids = build_corner_ids(llh, bbox_size_m)
        
        objects_with_ids.append({
            "spatial_ids": ids,
            "name": name,
            "confidence": confidence,
            "llh": llh,
            "timestamp": datetime.datetime.now().isoformat(),
        })
    return objects_with_ids

def lon_to_x(lon: float, zoom: int) -> int:
    """
    Convert longitude to x index at a given zoom level.
    """
    n = 1 << zoom               # 2^zoom
    return int(math.floor(n * (lon + 180.0) / 360.0))

def lat_to_y(lat: float, zoom: int) -> int:
    """
    Convert latitude to y index at a given zoom level.
    """
    lat = max(min(lat,  85.0511287798066), -85.0511287798066)  # clamp
    n = 1 << zoom
    lat_rad = math.radians(lat)
    return int(math.floor(
        n / 2.0 * (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi)
    ))


def alt_to_f(alt_m: float, zoom: int) -> int:
    """
    Convert altitude [m] to 'f' index at given zoom.
    At zoom 25 => 1 m resolution (f=height[m]).
    """
    n = 1 << zoom
    H_GLOBAL_M   = 2 ** 25             # definition used by Spatial-ID spec
    return int(math.floor(n * alt_m / H_GLOBAL_M))


def get_approximate_bbox_minmax(llh, bbox_size):
    """
    Calculate the minimum and maximum coordinates of the bounding box
    without transforming to Web Mercator and back.
    """
    # center coordinates
    lon, lat, alt = llh

    # width, depth, height
    dx, dy, dz = bbox_size 

    # Convert width, depth into lon/lat degrees
    lat_deg = dy / 111320.0  # 1 degree latitude ~ 111.32 km
    lon_deg = dx / (111320.0 * math.cos(math.radians(lat)))  # 1 degree longitude varies with latitude

    min_llh = ( lon - lon_deg / 2.0, lat - lat_deg / 2.0, FLOOR_HEIGHT)
    max_llh = ( lon + lon_deg / 2.0, lat + lat_deg / 2.0, dz + FLOOR_HEIGHT)
    
    return min_llh, max_llh

def get_bbox_minmax(llh, bbox_size):
    """
    Caliculate the minimum and maximum coordinates of the bounding box
    """
    # Transformers
    _T_LLH2WEBM = Transformer.from_crs("EPSG:4979", "EPSG:3857", always_xy=True)
    _T_WEBM2LLH = Transformer.from_crs("EPSG:3857", "EPSG:4979", always_xy=True)

    # center coordinates
    lon, lat, alt = llh
    cx, cy = _T_LLH2WEBM.transform(lon, lat)

    # width, depth, height
    dx, dy, dz = bbox_size 

    # Align all bounding box coordinates to the floor
    min_xyz =(cx - dx / 2.0, cy - dy / 2.0, FLOOR_HEIGHT)
    max_xyz = (cx + dx / 2.0, cy + dy / 2.0, dz + FLOOR_HEIGHT)

    # Back to lon/lat/alt
    min_ll = _T_WEBM2LLH.transform(min_xyz[0], min_xyz[1])
    max_ll = _T_WEBM2LLH.transform(max_xyz[0], max_xyz[1])

    min_llh = (min_ll[0], min_ll[1], min_xyz[2])  
    max_llh = (max_ll[0], max_ll[1], max_xyz[2])  
    return min_llh, max_llh

def biuld_ids(llh, bbox_size):
    
    #current_time = time.time()
    #bbox_min, bbox_max = get_bbox_minmax(llh, bbox_size)
    """ Use approximate bounding box calculation for performance."""
    bbox_min, bbox_max = get_approximate_bbox_minmax(llh, bbox_size)
    #print(f"bbox calculation time: {time.time() - current_time:.3f} seconds")

    result = []
    for zoom in TARGET_ZOOM_LEVELS:
        
        x_start = lon_to_x(bbox_min[0], zoom)
        x_end = lon_to_x(bbox_max[0], zoom)

        # In Web Mercator the Y axis increases southward, so use maxLat -> minLat.
        y_start = lat_to_y(bbox_max[1], zoom)
        y_end = lat_to_y(bbox_min[1], zoom)

        f_start = alt_to_f(bbox_min[2], zoom)
        f_end = alt_to_f(bbox_max[2], zoom)
    
        ids = []
        for x in range (x_start, x_end + 1):
            for y in range (y_start, y_end + 1):
                for f in range (f_start, f_end + 1):
                    ids.append(f"{zoom}/{x}/{y}/{f}")
        result.append({
            "zoom": zoom,
            "ids": ids
        })
    return result
        
def build_corner_ids(llh, bbox_size):
    """
    Build spatial IDs for the corners of the bounding box.
    """
    """ Use approximate bounding box calculation for performance."""
    bbox_min, bbox_max = get_approximate_bbox_minmax(llh, bbox_size)
    
    result = []

    for zoom in TARGET_ZOOM_LEVELS:
        
        x_start = lon_to_x(bbox_min[0], zoom)
        x_end = lon_to_x(bbox_max[0], zoom)

        # In Web Mercator the Y axis increases southward, so use maxLat -> minLat.
        y_start = lat_to_y(bbox_max[1], zoom)
        y_end = lat_to_y(bbox_min[1], zoom)

        f_start = alt_to_f(bbox_min[2], zoom)
        f_end = alt_to_f(bbox_max[2], zoom)

        result.append({
            "zoom": zoom,
            "min_corner": f"{zoom}/{x_start}/{y_start}/{f_start}",
            "max_corner": f"{zoom}/{x_end}/{y_end}/{f_end}"
        })
    return result