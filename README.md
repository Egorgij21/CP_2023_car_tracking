# CP_2023_car_tracking

## Steps to run Code


- Goto the cloned folder.
```
cd YOLOv8-DeepSORT-Object-Tracking
```
- Install the dependecies
```
pip install -e '.[dev]'

```

- Setting the Directory.
```
cd ultralytics/yolo/v8/detect

```

- For yolov8 object detection + Tracking + Vehicle Counting
```
python predict.py model=yolov8l.pt source="test3.mp4" show=False
```
