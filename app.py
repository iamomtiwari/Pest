import gdown
import onnxruntime as ort
import numpy as np

# Google Drive file ID (replace with your own file ID)
# Example link: https://drive.google.com/file/d/14iXBFxUBU8NidO3hTxdPnnpfuqWRCo66/view
file_id = "https://drive.google.com/file/d/1zLtrp6MS5SckC1JIld0EIJ9d9NtzpXkf/view?usp=sharing"
url = f'https://drive.google.com/uc?id={file_id}'
output = 'vit_model.onnx'

# Download ONNX model from Google Drive
gdown.download(url, output, quiet=False)

# Load the ONNX model
ort_session = ort.InferenceSession('vit_model.onnx')

# Define an example input (adjust dimensions based on your model)
dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)

# Run the model with the dummy input
outputs = ort_session.run(None, {'input': dummy_input})

# Print the model outputs (for testing)
print(outputs)
from PIL import Image
import torchvision.transforms as transforms

# Function to load and preprocess the image
def preprocess_image(image_path):
    input_size = (224, 224)  # Adjust this based on your model input size
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0).numpy()  # Add batch dimension
    return image.astype(np.float32)

# Modify this part to accept an image path
image_path = 'path_to_your_image.jpg'  # Replace with the actual image path
image_input = preprocess_image(image_path)

# Run the model with the actual image input
outputs = ort_session.run(None, {'input': image_input})
print(outputs)
