from turtle import st
from fastapi import APIRouter, File, Request
from ..utils.facemask_utils import prediction_mobilenet, prediction_yolov5
import torch
import face_detection
from torchvision.models import mobilenet_v2
from ..config import settings
from ..config import CONFIG

from pydantic import BaseModel, EmailStr

class ImageBytesIn(BaseModel):
    #text: str
    image: bytes

router = APIRouter(
    prefix="/facemask",
    tags=["facemask"],
    responses={404: {"description": "Not found"}},
)

@router.on_event("startup")
async def startup_event():
    """Model should be loaded before the application starts
    """
    print(torch.cuda.is_available())

    print(torch.cuda.device_count())

    global model
    # model = mobilenet_v2(pretrained=True)

    # model.classifier = torch.nn.Sequential(
    #         torch.nn.Dropout(p=0.2), 
    #         torch.nn.Linear(in_features=model.classifier[1].in_features, out_features=256),
    #         torch.nn.ReLU(),
    #         torch.nn.Linear(in_features=256, out_features=128),
    #         torch.nn.ReLU(),
    #         torch.nn.Linear(in_features=128, out_features=3)
    #         )

    # model.load_state_dict(torch.load(CONFIG['MODEL_PATH'], map_location=torch.device('cpu')))
    # print("Model loaded")
    # model.eval()

    # face detection
    global detector
    # detector = face_detection.build_detector("RetinaNetMobileNetV1", confidence_threshold=.65, nms_iou_threshold=.3) #face detection

    global model_yolo
    model_yolo = torch.hub.load('ultralytics/yolov5', 'custom', path=CONFIG['PATH_YOLOV5_FACEMASK'], device=torch.device('cuda:0')) 
    # model_yolo.conf = 0.2  # NMS confidence threshold
    # model_yolo.iou = 0.45  # NMS IoU threshold
    model_yolo.eval()

# @router.post("/testImage", response_model=ImageBytesIn)
# async def get_image(file: bytes = File(...)):
#     return file

class FacemaskInferenceOutput(BaseModel):
    label: str
    x0: int
    y0: int
    x1: int
    y1: int

@router.post("/classifier")
async def get_prediction_mobilenet(file: bytes = File(...)):
    """ the function is to get the image from the frontend
        and pass those image which is in bytes into another function
        for prediction

    Args:
        file (bytes, optional): The image from frontend is in
        bytes. Defaults to File(...).

    Returns:
        dict: output will be sent to frontend in dict
    """
    output = prediction_mobilenet(file, model, detector)

    #if isinstance(output, list): print("OLA ITS LIST!")

    return output

@router.post("/classifier/yolov5")
async def get_prediction_yolov5(file: bytes = File(...)):
    """ the function is to get the image from the frontend
        and pass those image which is in bytes into another function
        for prediction

    Args:
        file (bytes, optional): The image from frontend is in
        bytes. Defaults to File(...).

    Returns:
        dict: output will be sent to frontend in dict
    """
    output = prediction_yolov5(file, model_yolo)

    #if isinstance(output, list): print("OLA ITS LIST!")

    return output


