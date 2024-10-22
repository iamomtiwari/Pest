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
