from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo11m.pt")

    model.train(
        data="/media/rmedu-4090/New Volume1/samin-workforceRTX/license_plate_detector/config.yaml",
        epochs=500,
        plots=True,
        imgsz=640,
        batch=16,
        fliplr=0.0,
        mosaic=0.4,
        name="bangla-char-detector",
        workers=0
    )
