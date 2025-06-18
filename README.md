# Interior_Spatial_ID

## Setup  

**Only `transformation.py` and `main.py` is needed to run this project.**  
`stream.py` is used to check the output(RGB and 3 kinds of coordinates)

### Get Transformation Matrix

1. Stick the apriltag on the wall, manually set the `TAG_SIZE_M`, `TAG_LAT_DEG`, `TAG_LON_DEG`, `TAG_ALT_M` and `TAG_YAW_DEG` in **`transformation.py`**.  
   
2. Run **`transformation.py`**, it will print the camera coordinates, you can check the result to see if it's coorect. The matrix will be saved and you don't need to worry about it anymore.

### Run the Service

1. Run **`main.py`**.

### For Debug

1. Run **`stream.py`**.