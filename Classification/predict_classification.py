import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the STNResNet18 model (same as in training)
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
        x = F.grid_sample(x, grid)
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

# Load the saved model
num_classes = 4  
model = STNResNet18(num_classes).to(device)
model.load_state_dict(torch.load("STN_resnet_model.pth", map_location=device))
model.eval()

# Define data transformations (same as training)
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to predict an image
def predict_image(image_path, model, device):

    image = Image.open(image_path).convert("RGB")
    image = data_transforms(image).unsqueeze(0)  

    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = outputs.max(1)

    return predicted.item()

# Test the prediction function
image_path = "/home/hnad/acne_analysis/Classification/levle3_0.jpg"  
predicted_class = predict_image(image_path, model, device)

# Print the predicted class
print(f"Predicted class: {predicted_class}")
