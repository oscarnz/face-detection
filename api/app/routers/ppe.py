from fastapi import APIRouter, File
from ..utils.ppe_utils import prediction_ppe
import torch
from ..config import settings
from ..config import CONFIG

router = APIRouter(
    prefix="/ppe",
    tags=["ppe"],
    responses={404: {"description": "Not found"}},
)

@router.on_event("startup")
async def startup_event():
    """Model should be loaded before the application starts
    """
    global model_yolo
    model_yolo = torch.hub.load('ultralytics/yolov5', 'custom', path=CONFIG['PATH_YOLOV5_PPE'], device=torch.device('cuda:0')) 
    
    model_yolo.eval()

@router.post("/classifier")
async def get_prediction_ppe(file: bytes = File(...)):
    """the function is to get the image from the frontend
        and pass those image which is in bytes into another function
        for prediction

    Args:
        file (bytes, optional): The image from frontend is in
        bytes. Defaults to File(...).

    Returns:
        dict: output will be sent to frontend in dict
    """

    output = prediction_ppe(file, model_yolo)

    return output