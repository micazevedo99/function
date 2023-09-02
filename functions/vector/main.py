import os
import base64
import io
import json
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import torchvision
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, input_size=224, embedding_size=2048, pretrained=False):
        super(AlexNet, self).__init__()
        self.model = torchvision.models.alexnet(pretrained=pretrained)
        self.model.classifier = nn.Linear(
            in_features=9216, out_features=embedding_size, bias=True)

    def forward(self, x):
        return self.model(x)


def init_context(context):
    # Constants
    IMAGE_SIZE = 320
    EMBEDDING_SIZE = 2048
    # Read the model path from environment variable
    MODEL_PATH = os.environ["MODEL_PATH"]

    # Initialize the model
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    context.model = AlexNet(input_size=IMAGE_SIZE,
                            embedding_size=EMBEDDING_SIZE).to(DEVICE)
    context.model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    context.model.eval()

    # Define the image transform
    context.transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
    ])
    context.device = DEVICE


def handler(context, event):
    try:
        # Decode the Base64 image
        image_data = event.body.decode('utf-8')
        image_data = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_data)).convert('RGB')

        # Transform and process the image
        image_tensor = context.transform(image).unsqueeze(0).to(context.device)

        # Pass through the model
        with torch.no_grad():
            embedding = context.model(image_tensor)

        # Convert the tensor to a NumPy array
        feature_vector_np = embedding.cpu().numpy().ravel()

        # Normalize the vector using L2 norm
        feature_vector_normalized = feature_vector_np / \
            np.linalg.norm(feature_vector_np)

        return context.Response(body=json.dumps({"vector": feature_vector_normalized.tolist()}),
                                headers={},
                                content_type='application/json',
                                status_code=200)
    except Exception as e:
        return context.Response(body=json.dumps({"error": str(e)}),
                                headers={},
                                content_type='application/json',
                                status_code=500)
