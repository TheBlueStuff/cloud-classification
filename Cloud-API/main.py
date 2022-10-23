from platform import architecture
from fastapi import FastAPI, File, UploadFile
from typing import Optional, List
from typing import Union
import cv2
import numpy as np

from metadata import *
from inference import *
from models import *

app = FastAPI()
model = CNN()

@app.get("/")
def read_root():
    return({"hello": "world"})


@app.post("/singleFile")
async def create_file(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8) 
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cloud_type, image = infer(model.model, model.transform, img)
    return {
        "class": classes[5][1], 
        "description": descriptions[5],
        "image_url" : 'https://www.google.com/search?q=cloud&source=lnms&tbm=isch&sa=X&ved=2ahUKEwijqZmRhvf6AhWNErkGHRKOC_4Q_AUoAXoECAIQAw&biw=1920&bih=937&dpr=1#imgrc=Zfs7c0VnTZZXeM'
    }

@app.post("/multipleFiles")
async def create_file(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8) 
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cloud_type, image = infer(model.model, model.transform, img)
    return {
        "class": classes[5][1]
    }
