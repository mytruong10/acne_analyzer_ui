import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from ultralytics import YOLO
from PIL import Image
import cv2  
import numpy as np
import os
from find_max import A_max

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the STNResNet18 model
class STN(nn.Module):
    def __init__(self):
        super(STN, self).__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        with torch.no_grad():
            input_tensor = torch.zeros(1, 3, 224, 224)
            self.localization_output_size = self._get_localization_output_size(input_tensor)

        self.fc_loc = nn.Sequential(
            nn.Linear(self.localization_output_size, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def _get_localization_output_size(self, x):
        x = self.localization(x)
        return x.numel() // x.shape[0]

    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, self.localization_output_size)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid, align_corners=True)  
        return x

class STNResNet18(nn.Module):
    def __init__(self, num_classes):
        super(STNResNet18, self).__init__()
        self.stn = STN()
        self.resnet18 = models.resnet18(pretrained=True)
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, num_classes)

    def forward(self, x):
        x = self.stn(x)
        return self.resnet18(x)

# Load the YOLO model
yolo_model = YOLO('yolo.pt')

# Load the STN-ResNet18 model
num_classes = 4  
classification_model = STNResNet18(num_classes).to(device)
classification_model.load_state_dict(torch.load("STN_resnet_model.pth", map_location=device))
classification_model.eval()

# Define data transformations for classification model
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to calculate the normalized acne size as a percentage
def calculate_normalized_acne_size(acne_sizes):
    if len(acne_sizes) == 0 or A_max == 0:  
        return 0
    total_area = sum(width * height for width, height in acne_sizes)
    normalized_size = (total_area / A_max) * 100  
    return min(max(normalized_size, 0), 100), total_area  

# Function to map classification severity to percentage
def map_classification_to_percentage(classification_severity):
    severity_mapping = {0: 25, 1: 50, 2: 75, 3: 100}
    return severity_mapping.get(classification_severity, 25)  

# Function to calculate the final acne grade based on the combined score
def calculate_acne_grade_combined(normalized_acne_size, classification_percentage):
    combined_score = (normalized_acne_size + classification_percentage) / 2  
    print(f"Normalized Acne Size: {normalized_acne_size:.2f}%, Classification Percentage: {classification_percentage}%, Combined Score: {combined_score:.2f}%")

    T1, T2, T3 = 25, 50, 75
    if combined_score <= T1:
        return "Mild", combined_score
    elif T1 < combined_score <= T2:
        return "Moderate", combined_score
    elif T2 < combined_score <= T3:
        return "Severe", combined_score
    else:
        return "Very Severe", combined_score

# Function to perform grading based on YOLO detection and classification
def grade_acne(image_path, yolo_model, classification_model, device):
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}. Please check the file path.")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)

    input_tensor = data_transforms(image_pil).unsqueeze(0).to(device)

    results = yolo_model(image_path)

    num_acne = 0
    acne_sizes = []

    for result in results:
        boxes = result.boxes
        num_acne = len(boxes)

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            width = x2 - x1
            height = y2 - y1

            if width <= 0 or height <= 0:
                print(f"Invalid width or height for box: {box.xyxy[0]} (Width: {width}, Height: {height})")
                continue
            
            acne_sizes.append((width, height))

    normalized_acne_size, total_area = calculate_normalized_acne_size(acne_sizes)

    with torch.no_grad():
        classification_output = classification_model(input_tensor)
        _, classification_severity = classification_output.max(1)

    classification_percentage = map_classification_to_percentage(classification_severity.item())
    classification_result = classification_severity.item()  

    print(f"Acne Count: {num_acne}, Total Area: {total_area}, Classification Result: {classification_result}")

    grading, combined_score = calculate_acne_grade_combined(normalized_acne_size, classification_percentage)

    return grading

# Function to process all images in a folder
def process_images_in_folder(folder_path, yolo_model, classification_model, device):
    if not os.path.exists(folder_path):
        raise ValueError(f"The folder {folder_path} does not exist.")
    
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        print(f"Processing image: {image_path}")
        try:
            grading, combined_score = grade_acne(image_path, yolo_model, classification_model, device)
            print(f"Grading for {image_file}: {grading}, Combined Score: {combined_score:.2f}%")
        except Exception as e:
            print(f"Error processing {image_file}: {e}")

# Example usage
#test_image_folder = '/home/hnad/classification/Test_image'
#process_images_in_folder(test_image_folder, yolo_model, classification_model, device)
