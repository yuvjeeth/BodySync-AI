from ultralytics import YOLO

if __name__ == '__main__':
    # hyperparameter tuning
    model = YOLO("yolov8n.pt")

    model.tune(
        data="data.yaml",
        epochs=30,
        iterations=20,
        imgsz=640,
        batch=8,
        workers=0,
        optimizer="SGD"
    )

    # train final model
    best = YOLO("runs/detect/tune/best.pt")

    best.train(
        data="data.yaml",
        epochs=50,
        batch=8,
        workers=0,
        name="final_tuned"
    )

    # evaluate tuned model
    model = YOLO("runs/detect/final_tuned/weights/best.pt")
    metrics = model.val()

    print(metrics)

    # test image (optional)
    # model("test.jpg", show=True)