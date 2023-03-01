from platform import architecture
from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import Optional, List
from typing import Union
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np

from metadata import *
from inference import *
from models import *

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
model = CNN()

@app.get("/")
def read_root():
    return({"hello": "world"})


@app.post("/singleFile")
async def create_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.fromstring(contents, np.uint8) 
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        cloud_type, image, prob = infer(model.model, model.transform, img)
        return {
            "class": classes[cloud_type][1], 
            "description": descriptions[cloud_type],
            "image" : image,
            "prob" : prob
        }
    except Exception as e:
        raise HTTPException(status_code=418, detail="wrong input")

@app.post("/multipleFiles")
async def create_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.fromstring(contents, np.uint8) 
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cloud_type = infer_multiple(model.model, model.transform, img)
        return {
            "file": file.filename,
            "class": classes[cloud_type][1]
        }
    except Exception as e:
        raise HTTPException(status_code=418, detail="wrong input")
