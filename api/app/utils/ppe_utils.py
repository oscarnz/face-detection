from PIL import Image
import numpy as np  
import io

def prediction_ppe(binary_image, model_yolo):
  """This function gets image binary to transform to detect the person, the 
     color of the helmet and vest.

  Args:
      binary_image (bytes): Image properties in bytes (from frontend)

  Returns:
      responses (dict): the result of the prediction for detected object
                        will be in dictionary and will be passed to frontend
                        as json
  """

  PPE_LABELS = {
    0: "person",
    1: "vest",  
    2: "blue helmet",   
    3: "red helmet", 
    4: "white helmet",
    5: "yellow helmet"
}
  input_image = Image.open(io.BytesIO(binary_image)).convert("RGB")
  img = np.uint8(input_image)
  
  output = model_yolo(img) #object detection on image
  #print(list(output.xyxy[0]))

  #print(output)

  response = []

  for object in list(output.xyxy[0]):
      x1 = int(object[0]) #x1
      y1 = int(object[1]) #y1
      x2 = int(object[2]) #x2
      y2 = int(object[3]) #y2
      label = int(object[5]) #label for each class of ppe
      label_name = PPE_LABELS[int(label)]

      resp = {}
      resp["label"] = str(label_name)
      resp["x1"] = str(x1)
      resp["y1"] = str(y1)
      resp["x2"] = str(x2)
      resp["y2"] = str(y2)
      response.append(resp)

      # print(x1, y1, x2, y2, label)

  return response