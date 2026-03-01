import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

CLASS_NAMES = ['Benign', 'Malignant', 'Normal']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def load_model(model_path: str):
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(2048, 3)
    )
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def predict(image_path: str, model) -> dict:
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_idx = probabilities.argmax().item()

    return {
        "prediction": CLASS_NAMES[predicted_idx],
        "confidence": round(probabilities[predicted_idx].item() * 100, 2),
        "all_scores": {
            CLASS_NAMES[i]: round(probabilities[i].item() * 100, 2)
            for i in range(3)
        }
    }