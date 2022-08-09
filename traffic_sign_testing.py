from cProfile import label
import numpy as np
import cv2
import pickle
import keras
import skimage.morphology as morp
from skimage.filters import rank
from sklearn.utils import shuffle

width= 640         # CAMERA RESOLUTION
height = 480
brightness = 180
threshold = 0.90        # PROBABLITY THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX

#camera launch 
cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
cap.set(cv2.CAP_PROP_FPS,30)
cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))

model =keras.models.load_model('traffice_sign_classification.h5')


def gray_scale(img):
  img= cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  return img

def image_normalize(img):
   img = np.divide(img, 255)
   return img
   
def local_histo_equalize(image):
    kernel = morp.disk(30)
    img= rank.equalize(image, selem=kernel)
    return img

def preprocessing(img):
    img = gray_scale(img)
    img = image_normalize(img)
    img=local_histo_equalize(img)
    img = img/255
    return img


def getClassname(image_labels):
    if  image_labels == 0: 
      return 'Speed Limit 20 km/h'
    elif image_labels == 1:
       return 'Speed Limit 30 km/h'
    elif image_labels == 2: 
      return 'Speed Limit 50 km/h'
    elif image_labels == 3:
       return 'Speed Limit 60 km/h'
    elif image_labels == 4: 
      return 'Speed Limit 70 km/h'
    elif image_labels == 5: 
      return 'Speed Limit 80 km/h'
    elif image_labels == 6: 
      return 'End of Speed Limit 80 km/h'
    elif image_labels == 7: 
      return 'Speed Limit 100 km/h'
    elif image_labels == 8: 
      return 'Speed Limit 120 km/h'
    elif image_labels == 9: 
      return 'No passing'
    elif image_labels == 10: 
      return 'No passing for vechiles over 3.5 metric tons'
    elif image_labels == 11: 
      return 'Right-of-way at the next intersection'
    elif image_labels == 12: 
      return 'Priority road'
    elif image_labels == 13: 
      return 'Yield'
    elif image_labels == 14: 
      return 'Stop'
    elif image_labels == 15:
       return 'No vechiles'
    elif image_labels == 16: 
      return 'Vechiles over 3.5 metric tons prohibited'
    elif image_labels == 17: 
      return 'No entry'
    elif image_labels == 18: 
      return 'General caution'
    elif image_labels == 19: 
      return 'Dangerous curve to the left'
    elif image_labels == 20: 
      return 'Dangerous curve to the right'
    elif image_labels == 21: 
      return 'Double curve'
    elif image_labels == 22: 
      return 'Bumpy road'
    elif image_labels == 23:
      return 'Slippery road'
    elif image_labels == 24: 
      return 'Road narrows on the right'
    elif image_labels == 25: 
      return 'Road work'
    elif image_labels == 26: 
      return 'Traffic signals'
    elif image_labels == 27: 
      return 'Pedestrians'
    elif image_labels == 28: 
      return 'Children crossing'
    elif image_labels == 29: 
      return 'Bicycles crossing'
    elif image_labels == 30: 
      return 'Beware of ice/snow'
    elif image_labels == 31: 
      return 'Wild animals crossing'
    elif image_labels == 32: 
      return 'End of all speed and passing limits'
    elif image_labels == 33: 
      return 'Turn right ahead'
    elif image_labels == 34: 
      return 'Turn left ahead'
    elif image_labels == 35: 
      return 'Ahead only'
    elif image_labels == 36: 
      return 'Go straight or right'
    elif image_labels == 37: 
      return 'Go straight or left'
    elif image_labels == 38:
       return 'Keep right'
    elif image_labels == 39: 
      return 'Keep left'
    elif image_labels == 40:
       return 'Roundabout mandatory'
    elif image_labels == 41:
       return 'End of no passing'
    elif image_labels == 42:
       return 'End of no passing by vechiles over 3.5 metric tons'

while True:
 
  # READ IMAGE
  ignore, frame = cap.read()  
  
  # PROCESS IMAGE
  img =np.asarray(frame)
  img =cv2.resize(img,(32,32))
  img =preprocessing(img)
  cv2.imshow("Processed Image", img)
  img =img.reshape(1, 32, 32,  1 )

  #predict 
  predictions=model.predict(img)          #returns prediction score 
  class_index=np.argmax(predictions,axis=1)  #returns class no 
  probability=np.amax(predictions)
  print(class_index)

  cv2.putText(frame, "CLASS: " , (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
  cv2.putText(frame, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

  if probability > threshold:   #print(getCalssName(classIndex))
    
    cv2.putText(frame,str(class_index)+" "+str(getClassname(class_index)), (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, str(round(probability*100,2) )+"%", (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    
  cv2.imshow("Result",frame)
  
  if cv2.waitKey(1) and 0xFF == ord('q'):
    break
cap.release()