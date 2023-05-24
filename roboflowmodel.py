from roboflow import Roboflow
rf = Roboflow(api_key="mtMWsmfjWMmb8VaZeSpe")
project = rf.workspace().project("baseball-detection-2")
model = project.version(1).model

# infer on a local image
print(model.predict("your_image.jpg", confidence=40, overlap=30).json())