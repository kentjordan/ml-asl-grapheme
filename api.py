
from fastapi import FastAPI, UploadFile
from PIL import Image

import io

import torch
from torchvision.transforms import Resize, ToTensor, Compose

from datasets import ASLTrainDataset
from model import ASLModel

train_dataset = ASLTrainDataset("./dataset/asl_alphabet_train")
model = ASLModel()
model.load_state_dict(torch.load("./models/dev.pt"))

app = FastAPI()

@app.post('/predict/{w}/{h}')
async def predict(photo: UploadFile, w: int = 200, h: int = 200):

    image = Image.open(io.BytesIO(await photo.read()))
    
    transforms = Compose([
        Resize(size=(w, h)),
        ToTensor()
    ])
    image = transforms(image)
    image = torch.unsqueeze(image, 0) # type: ignore
    output = model(image)
    prediction = train_dataset.classes[torch.argmax(output)]

    return {
        "prediction": prediction
    }
