import logging
from ResNet import resnet18
from PIL import Image
from io import BytesIO
from torch import Tensor
import torch
from torchvision import transforms
import base64
import torch.nn.functional as F


def load_model(model_path = None, num_classes = 2):
    model =resnet18(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model
    

def open_image(image: str, applicatin: str) -> Tensor:
    means, stds = None, None
    if applicatin == "mangoes-leaves1":
        means = torch.tensor([0.2399, 0.2776, 0.2590])
        stds = torch.tensor([0.0703, 0.0997, 0.0848])
    if applicatin == "mangoes-quality":
        means = torch.tensor([0.7322, 0.6320, 0.5282])
        stds = torch.tensor([0.2886, 0.3069, 0.3414])

    if applicatin == "mangoes-species":
        means =  torch.tensor([0.5191, 0.4742, 0.4206])
        stds = torch.tensor([0.2083, 0.2137, 0.2235])
    if applicatin == "maize-diseases":
        means = torch.tensor([0.4389, 0.4967, 0.3800])
        stds = torch.tensor([0.1792, 0.1678, 0.1748])
    
    if means is None or stds is None:
        raise ValueError("Unknown application {}".format(applicatin))
    transformations = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean = means, std = stds)
    ])

    image_bytes = image.encode('utf-8')
    image = Image.open(BytesIO(base64.b64decode(image_bytes))).convert(mode="RGB")
    input_tensor = transformations(image).unsqueeze(0)
    input_tensor = input_tensor.to('cpu')
    return input_tensor

def predict(image: str, applicatin: str, model = None) -> str:

    input_tensor = open_image(image, applicatin)
    output = model(input_tensor)

    probabilities = F.softmax(output, dim=1)
    probability, _class = probabilities.topk(1, dim = 1)
    confidence, top_label = probability.tolist()[0][0], _class.tolist()[0][0]


    # Compose the results
    return {'confidence': confidence, 'results': top_la
