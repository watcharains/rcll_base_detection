# imports 
import cv2 as cv 
import numpy as np
from pyzbar.pyzbar import decode
import pyzbar
import time
import AiPhile 
import re
import tensorflow.lite as tflite
from PIL import Image
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 640


def load_labels(label_path):
    r"""Returns a list of labels"""
    with open(label_path) as f:
        labels = {}
        for line in f.readlines():
            m = re.match(r"(\d+)\s+(\w+)", line.strip())
            labels[int(m.group(1))] = m.group(2)
        return labels
def load_model(model_path):
    r"""Load TFLite model, returns a Interpreter instance."""
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def getAvailableCameraIds(max_to_test):
    available_ids = []
    for i in range(max_to_test):
        temp_camera = cv.VideoCapture(i)
        if temp_camera.isOpened():
            temp_camera.release()
            print("found camera with id {}".format(i))
            available_ids.append(i)
    return available_ids

def detectQRcode(image):
    # convert the color image to gray scale image
    Gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # create QR code object
    objectQRcode = pyzbar.pyzbar.decode(Gray)
    for obDecoded in objectQRcode: 
        x, y, w, h =obDecoded.rect
        cv.rectangle(image, (x,y), (x+w, y+h), ORANGE, 4)
        points = obDecoded.polygon
        #if len(points) > 4:
        #    hull = cv.convexHull(
         #       np.array([points for point in points], dtype=np.float32))
        #    hull = list(map(tuple, np.squeeze(hull)))
        #else:
        hull = points
  
        return hull



model_path = 'ei-ired_new_bobobox-object-detection-tensorflow-lite-float32-model.lite'
#model_path = 'ei-ired_new_bobobox-object-detection-tensorflow-lite-int8-quantized-model.lite'
label_path = 'label.txt'

#getAvailableCameraIds(10)

cap = cv.VideoCapture(0,cv.CAP_DSHOW)
#cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FOURCC,  1196444237)
cap.set(3,640)
cap.set(4,640)
cap.set(cv.CAP_PROP_FPS, 30)
ret, frame = cap.read()
image_width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
image_height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
print(ret)
print(frame)


frame_counter =0
starting_time =time.time()

interpreter = load_model(model_path)
labels = load_labels(label_path)

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
width = input_details[0]['shape'][2]
height = input_details[0]['shape'][1]

# Get input index
input_index = input_details[0]['index']

while True:
    ret, frame = cap.read()

    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    #frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_resized = cv.resize(frame_rgb, (width, height))
    frame_resized = frame_resized.astype(np.float32)
    frame_resized /= 255.
    #frame_resized = frame_resized.astype(np.int8)
    #frame_resized /= 255.
    input_data = np.expand_dims(frame_resized, axis=0)
    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    #if floating_model:
    #    input_data = (np.float32(input_data) - input_mean) / input_std

    # set frame as input tensors
    interpreter.set_tensor(input_details[0]['index'], input_data)
    # perform inference
    interpreter.invoke()
    # Get output tensor
    output_details = interpreter.get_output_details()
    # print(output_details)
    # output_details[0] - position
    # output_details[1] - class id
    # output_details[2] - score
    # output_details[3] - count

    #positions = np.squeeze(interpreter.get_tensor(output_details[0]['index']))
    #classes = np.squeeze(interpreter.get_tensor(output_details[1]['index']))
    #scores = np.squeeze(interpreter.get_tensor(output_details[2]['index']))

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[1]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[3]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[0]['index'])[0] # Confidence of detected objects

    #boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    #classes = interpreter.get_tensor(output_details[1]['index'])[0]
    #scores = interpreter.get_tensor(output_details[2]['index'])[0]
    #output = interpreter.get_output_details()[0]  # Model has single output.
    #sh = interpreter.get_tensor(output['index']).shape
    #image = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
    #image = image.resize((width, height))

   # top_result = process_image(interpreter, image, input_index)
   # display_result(top_result, frame, labels)

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    #print("-->",scores )
    for i in range(len(scores)):
        if ((scores[i] > 0.5) and (scores[i] <= 1.0)):
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1, (boxes[i][0] * image_height)))
            xmin = int(max(1, (boxes[i][1] * image_width)))
            ymax = int(min(image_height, (boxes[i][2] * image_height)))
            xmax = int(min(image_width, (boxes[i][3] * image_width)))

            cv.rectangle(frame, (xmin, ymin),
                          (xmax, ymax), (10, 255, 0), 4)

            # Draw label
            object_name = labels[int(classes[i])]
            label = '%s: %d%%' % (object_name, int(scores[i] * 100))
            labelSize, baseLine = cv.getTextSize(
                label, cv.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            # Make sure not to draw label too close to top of window
            label_ymin = max(ymin, labelSize[1] + 10)
            cv.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10), (
                xmin + labelSize[0], label_ymin + baseLine - 10), (255, 255, 255), cv.FILLED)
            cv.putText(frame, label, (xmin, label_ymin - 7),
                        cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    #if time.time() - start >= 1:
    #    print('fps:', frame_counter)
    ##    frame_counter = 0
    #    start = time.time()

    ##cv.imshow('Object detector', frame)



    Gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #Gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    # create QR code object
    objectQRcode = pyzbar.pyzbar.decode(Gray)
    for obDecoded in objectQRcode:
        x, y, w, h = obDecoded.rect
        cv.rectangle(frame, (x, y), (x + w, y + h), AiPhile.ORANGE, 4)

    #fps = frame_counter/(time.time()-starting_time)
    #AiPhile.textBGoutline(frame, f'FPS: {round(fps,1)}', (30,40), scaling=0.6)
    cv.imshow("image", frame)

    # Press 'q' to quit
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

# keep looping until the 'q' key is pressed
while True:
    frame_counter +=1
    ret, frame = cap.read()
    #print(ret)


#    hull_points =detectQRcode(frame)
#if hull_points:
 #       pt1, pt2, pt3, pt4 = hull_points
 #       frame =AiPhile.fillPolyTrans(frame, hull_points, AiPhile.MAGENTA, 0.6)
#        # AiPhile.textBGoutline(frame, f'Detection: Pyzbar', (30,80), scaling=0.5,text_color=(AiPhile.MAGENTA ))
 #       cv.circle(frame, pt1, 3, AiPhile.GREEN, 3)
 #       cv.circle(frame, pt2, 3, (255, 0, 0), 3)
##        cv.circle(frame, pt3, 3,AiPhile.YELLOW, 3)
#        cv.circle(frame, pt4, 3, (0, 0, 255), 3)

    Gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #Gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    # create QR code object
    objectQRcode = pyzbar.pyzbar.decode(Gray)
    for obDecoded in objectQRcode:
        x, y, w, h = obDecoded.rect
        cv.rectangle(frame, (x, y), (x + w, y + h), AiPhile.ORANGE, 4)

    key = cv.waitKey(1)
    if key == ord("q"):
        break
    fps = frame_counter/(time.time()-starting_time)
    AiPhile.textBGoutline(frame, f'FPS: {round(fps,1)}', (30,40), scaling=0.6)
    cv.imshow("image", frame)
# close all open windows
cv.destroyAllWindows() 
cap.release()