from ultralytics import YOLO

# Load the pre-trained model
model = YOLO('/home/hnad/acne_analysis/YOLO/runs/detect/train5/weights/best.pt')

# Perform inference
results = model('/home/hnad/acne_analysis/YOLO/levle1_3.jpg')

# Loop through the results and extract object count and sizes
for result in results:

    result.show()
    # Extract bounding boxes (x1, y1, x2, y2) and number of objects
    boxes = result.boxes  
    
    # Number of detected objects
    num_objects = len(boxes)
    print(f"Number of detected objects: {num_objects}")
    
    # Loop through each bounding box
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]  

        width = x2 - x1
        height = y2 - y1

        print(f"Object size - Width: {width}, Height: {height}")

