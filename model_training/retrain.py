from ultralytics import YOLO

data = 'baseball-detection-2.v3i.yolov8/data.yaml'
imgsz = 640
epochs = 150
batch_size = 16
name = "pitch_detection_v4"
model_path = 'runs/detect/pitch_detection_v12/weights/best.pt'

# Trains the model based on given parameters
def train_model(data, imgsz, epochs, batch_size, name, model_path):
    model = YOLO(model_path)
    print("Model loaded")
    model.train(
        data=data,
        imgsz=imgsz,
        epochs=epochs,
        batch=batch_size,
        name=name
    )
    print("Model trained")
    return model

if __name__ == "__main__":
    #train a model
    fine_tuned_model = train_model(data, imgsz, epochs, batch_size, name, 
                                   model_path)
    #export model
    fine_tuned_model.export(format='onnx')