import requests
from PIL import Image
from requests_toolbelt.multipart.encoder import MultipartEncoder
from io import BytesIO
from time import sleep, time


def process(image, server_url: str):
    """handle and read the images

    Args:
        image (_type_): _description_
        server_url (str): _description_

    Returns:
        _type_: _description_
    """
    start_time = time()
    m = MultipartEncoder(fields={"file": ("filename", image, "image/jpeg")})
    r = []
    #send input given by the user to fastAPI backend using Post method 
    r = requests.post(server_url, data=m, headers={"Content-Type": m.content_type}, timeout=8000)
    """send the input given by the user to the fastAPI backend using Post method

    Returns:
        _type_: _description_
    """
    
    rdict = r.json()
    total = time() - start_time
    print("http request " + str(total)) 
    """Convert into dictionary

    Returns:
        _type_: _description_
    """

    return str(rdict)

def process_video(image_str, server_url:str):
    start_time = time()
    m = MultipartEncoder(fields={"file": ("filename", image_str, "image/jpeg")})
    r = []
    r = requests.post(server_url, data=m, headers={"Content-Type": m.content_type}, timeout=8000)
    rdict = r.json()
    total = time() - start_time
    print("http request " + str(total)) 

    return str(rdict)

def load_image(image_file):
    """Open the images file

    Args:
        image_file (_type_): _description_
    """

    img = Image.open(image_file)
    
    return img

def load_video(frame):
    """Open the videos

    Args:
        frame (_type_): _description_

    Returns:
        _type_: _description_
    """
    start_time = time()
    img = Image.fromarray(frame)
    b = BytesIO()
    img.save(b,format="jpeg")
    total = time() - start_time
    print("convert image " + str(total)) 
    return b