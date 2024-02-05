from pydantic import BaseSettings

class Settings(BaseSettings):
    PATH_MOBILENET: str = "model/pytorch_mobilenet_5.pth"
    PATH_YOLOV5: str = "model/yolov5-300-epochs.pt"


GLOBAL_CONFIG = {
    "MODEL_PATH" : "./model/pytorch_mobilenet_5.pth",
    "PATH_YOLOV5_PPE" : "./model/best.pt",
    "PATH_YOLOV5_FACEMASK" : "./model/yolov5-100-epochs-3500-img.pt"
}

def get_config() -> dict:
    """
    Get config based on running environment
    :return: dict of config
    """

    config = GLOBAL_CONFIG.copy()
    return config

CONFIG = get_config()

settings = Settings()