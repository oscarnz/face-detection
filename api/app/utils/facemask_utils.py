from PIL import Image
from torchvision.transforms import transforms
import numpy as np  
import io
from time import sleep, time

FACEMASK_LABELS = {
  0: "Without Mask",
  1: "Incorrect Mask",  
  2: "With Mask"
}
#Bounding box color code for each object detection in PPE detection
LABEL_COLORS = {
    0: (55, 0, 0), # red
    1: (255, 255, 0), # yellow
    2: (0, 255, 0)  # green
}

#classify using mobilenet model
def classifier(image_tensor, model): 
  """Predict the image using loaded model

  Args:
      image_tensor (tensor): image properties in tensor

  Returns:
      pred (tensor): model prediction is in [1,1] tensor
  """
  output = model(image_tensor)
  pred=output.argmax(1)

  return pred
  
#latest version of face mask detection using retinanetmobilenet
def prediction_mobilenet(binary_image, model, detector):
  """This function gets image binary to transform and used in face detection model and 
      will be passed to facemask classifier to predict whether the extracted face is
      wearing mask or not. 

  Args:
      binary_image (bytes): Image properties in bytes (from frontend)
      model : loaded model from main.py

  Returns:
      responses (dict): the result of the prediction for each face will be 
                        in dictionary and will be passed to frontend
                        as json
  """
  input_image = Image.open(io.BytesIO(binary_image)).convert("RGB")
  img = np.uint8(input_image)
  #transform images
  prepare_image_seq = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
  #bounding boxes
  bounding_boxes = detector.detect(np.array(img))

  #probability for image predicted
  #print("Probabilities : ", bounding_boxes[:, 4])
  
  response = []
  # print(bounding_boxes[1])
  #bounding box
  for i in bounding_boxes[:, :4]:
    
    x0, y0, x1, y1 = [int(_) for _ in i]

    #place the coordinates of the bounding box into the image 
    face_to_analyze = img[max(11, y0)-10:y0+y1+10, max(11, x0)-10:x0+x1+10]
    #change the image into array
    face_to_analyze = Image.fromarray(np.uint8(face_to_analyze))
    #transform the array
    face_prepared = prepare_image_seq(face_to_analyze)
    #print(face_prepared)

    #feed the image tensor into the model 
    pred = classifier(face_prepared.unsqueeze(0), model)
    if pred == 0:
      label = "Without Mask"
    elif pred == 1:
      label = "Mask"
    else:
      label = "Incorrect Mask"
    
    #the result will be converted into dictionary and will be sent into frontend 
    resp = {}
    resp["label"] = label
    resp["x0"] = str(x0)
    resp["y0"] = str(y0)
    resp["x1"] = str(x1)
    resp["y1"] = str(y1)
    response.append(resp)

  return response

def prediction_yolov5(binary_image, model_yolo):
  """_summary_

  Args:
      binary_image (_type_): _description_
      model_yolo (_type_): _description_

  Returns:
      _type_: _description_
  """
  #start_time = time()
  input_image = Image.open(io.BytesIO(binary_image)).convert("RGB")
  img = np.uint8(input_image)
  #total = time() - start_time
  #print("process image " + str(total))

  #start_time_model = time()
  output = model_yolo(img, size=320) #object detection on image
  #total = time() - start_time_model
  #print("process model " + str(total)) 
  #print(output.xyxy[0])
  response = []

  for object in list(output.xyxy[0]):
      x1 = int(object[0]) #x1
      y1 = int(object[1]) #y1
      x2 = int(object[2]) #x2
      y2 = int(object[3]) #y2
      conf = round(float(object[4]),2)
      label = object[5] #label for each class of ppe
      label_name = FACEMASK_LABELS[int(label)]
      #label_color = LABEL_COLORS[int(label)]

      resp = {}
      resp["label"] = str(label_name)
      #resp["label_color"] = str(label_color)
      resp["x1"] = str(x1)
      resp["y1"] = str(y1)
      resp["x2"] = str(x2)
      resp["y2"] = str(y2)
      resp["confidence"] = str(conf)
      response.append(resp)
    

    # print(x1, y1, x2, y2, label)
  return response