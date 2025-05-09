from ultralytics import YOLO

# Load a model
model = YOLO("/home/hnad/acne_analysis/YOLO/best-06MAP.pt") 

# Use the model
model.train(data="/home/hnad/acne_analysis/YOLO/data-2/data.yaml", epochs=200)  
