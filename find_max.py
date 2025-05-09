import torch
from ultralytics import YOLO
import cv2
import os
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

yolo_model = YOLO('/home/hnad/acne_analysis/yolo.pt')

def calculate_acne_score(acne_sizes):
    return sum(width * height for width, height in acne_sizes)

def grade_acne(image_path, yolo_model):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}. Please check the file path.")

    results = yolo_model(image_path)
    acne_sizes = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            width = x2 - x1
            height = y2 - y1
            if width > 0 and height > 0:
                acne_sizes.append((width, height))

    acne_score = calculate_acne_score(acne_sizes)
    return acne_score

def process_images_and_find_max_score(folder_path, yolo_model, csv_writer):
    if not os.path.exists(folder_path):
        raise ValueError(f"The folder {folder_path} does not exist.")

    max_acne_score = float('-inf')  
    max_file_info = None          

    for root, _, files in os.walk(folder_path):
        folder_name = os.path.basename(root)  
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):  
                image_path = os.path.join(root, file)
                try:
                    acne_score = grade_acne(image_path, yolo_model)
                    csv_writer.writerow([folder_name, file, acne_score])  

                    if acne_score > max_acne_score:
                        max_acne_score = acne_score
                        max_file_info = (folder_name, file, max_acne_score)

                except Exception as e:
                    print(f"Error processing {file}: {e}")

    return max_file_info  

output_csv_file = '/home/hnad/acne_analysis/Acne_scores.csv'

with open(output_csv_file, mode='w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Folder Name', 'File Name', 'Acne Score'])  

    test_image_folder = '/home/hnad/acne_analysis/Dataset/JPEGImages'
    max_score_info = process_images_and_find_max_score(test_image_folder, yolo_model, csv_writer)

if max_score_info:
    folder_name, file_name, A_max = max_score_info
    print(f"Maximum acne score (A_max): {A_max} found in folder '{folder_name}' with file '{file_name}'.")
else:
    print("No valid images found.")

A_max = max_score_info[2] if max_score_info else 0
