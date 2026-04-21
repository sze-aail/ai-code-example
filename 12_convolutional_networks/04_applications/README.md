# 04_applications

## People Detection From Camera (Ultralytics)

File: `01_people_detection_camera.py`

This app runs realtime person detection (class `person`) on a camera stream or video file using Ultralytics YOLO.

### Quick run

```powershell
Set-Location "D:\hakiko_ai_ws\education\ai-code-example"
python "12_convolutional_networks\04_applications\01_people_detection_camera.py" --source 0
```

Press `q` to exit the window.

### Save annotated output video

```powershell
Set-Location "D:\hakiko_ai_ws\education\ai-code-example"
python "12_convolutional_networks\04_applications\01_people_detection_camera.py" --source 0 --save "runs\people_cam.mp4"
```

### Run on an existing video file

```powershell
Set-Location "D:\hakiko_ai_ws\education\ai-code-example"
python "12_convolutional_networks\04_applications\01_people_detection_camera.py" --source "sample.mp4" --save "runs\sample_people.mp4"
```

### Smoke test

```powershell
Set-Location "D:\hakiko_ai_ws\education\ai-code-example"
python "12_convolutional_networks\04_applications\smoke_test_people_detector.py"
```

