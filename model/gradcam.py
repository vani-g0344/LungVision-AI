import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
import base64
from io import BytesIO

# Transform — same as inference
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None

        # Hook into the last conv layer of ResNet50
        target_layer = model.layer4[2].conv3

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, image_path: str, class_idx: int) -> str:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        tensor = transform(image).unsqueeze(0)

        # Forward pass
        self.model.eval()
        output = self.model(tensor)

        # Backward pass for target class
        self.model.zero_grad()
        output[0][class_idx].backward()

        # Generate heatmap
        gradients = self.gradients[0]          # [C, H, W]
        activations = self.activations[0]      # [C, H, W]

        # Global average pool the gradients
        weights = gradients.mean(dim=(1, 2))   # [C]

        # Weighted sum of activations
        cam = torch.zeros(activations.shape[1:])
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU and normalize
        cam = torch.relu(cam)
        cam = cam.numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        # Resize to original image size
        cam_image = Image.fromarray((cam * 255).astype(np.uint8))
        cam_image = cam_image.resize((224, 224), Image.LANCZOS)
        cam_array = np.array(cam_image)

        # Create color heatmap (red=high, blue=low)
        heatmap = np.zeros((224, 224, 3), dtype=np.uint8)
        heatmap[:, :, 0] = cam_array                    # Red channel
        heatmap[:, :, 1] = (cam_array * 0.5).astype(np.uint8)  # Green
        heatmap[:, :, 2] = (255 - cam_array)            # Blue channel

        # Overlay heatmap on original image
        original = np.array(image.resize((224, 224)))
        overlay = (original * 0.6 + heatmap * 0.4).astype(np.uint8)

        # Convert to base64 string to send to frontend
        result_image = Image.fromarray(overlay)
        buffer = BytesIO()
        result_image.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()

        return f"data:image/png;base64,{img_str}"