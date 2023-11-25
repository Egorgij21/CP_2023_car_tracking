<!--Start-->
<div align="center">

  <img src="logo.png" alt="logo" width="200" height="auto" />
  <h1>🔥 Miss MISIS x AI Talent Hub team presents:</h1>
  
  <p>
    Car tracking and counting for Digital breakthrough 2023 
  </p>
  
</div>

<br />

# :notebook_with_decorative_cover: Table of Contents
<!-- Table of Contents -->
- [Building project](#compass-steps-to-run-code)
- [Setting up models](#gear-link-to-weights)
- [Inferencing video](#movie-camera-single-video-inference)

<!-- Building project -->
### 🧭 Steps to run Code

- Goto the cloned folder 
```
cd YOLOv8-DeepSORT-Object-Tracking
```
- Install the dependecies
```
pip install -e '.[dev]'
```

<!-- Setting up models -->
### ⚙️ Links to weights:
- [Deep Sort Onnx](https://disk.yandex.com/d/LN69wukpZystzg)

```
cp path-to-deep-sort/deep_sort.onnx YOLOv8-DeepSORT-Object-Tracking/ultralytics/yolo/v8/detect/
```

- [YOLOv8 fp16 cuda](https://disk.yandex.com/d/wml7Ti67_0-ikA)

```
cp path-to-yolov8/yolov8l_half.onnx YOLOv8-DeepSORT-Object-Tracking/ultralytics/yolo/v8/
```


<!-- Inferencing video -->
### 🎥 Single video inference
- Setting the Directory
```
cd ultralytics/yolo/v8/detect

```

- Необходимы видео source и полигон для видео в формате json
```
python predict.py source="$path1" json_path="$path2"
```
