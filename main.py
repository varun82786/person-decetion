""" Python Code to Recognise and Count No. of Person appeared in the Screen
	Sending or Uploading that Recognised Value into a channel in ThingSpeak-API
	Developed by Team Originators
	NAGA SAI GANESH, J L MANIKANTA, CH NAVEEN TEJA	"""
	
import cv2
import numpy as np
import time
import httplib2
import urllib
import http.client

key = 'V3QAOE5FG7N993IY'  # Thingspeak channel to update
temp=0

def webApi():
    while True:
        params = urllib.parse.urlencode({'field1': 17.443436, 'field2': 78.374243, 'field3': temp, 'key':key })
        headers = {"Content-typZZe": "application/x-www-form-urlencoded","Accept": "text/plain"}
        conn = http.client.HTTPConnection("api.thingspeak.com:80")
        try:
            conn.request("POST", "/update", params, headers)
            response = conn.getresponse()
            #print (temp);
            #print (response.status, response.reason);
            data = response.read()
            conn.close()
        except:
            print ("connection failed");
        return;



# Load Yolo
net = cv2.dnn.readNet("weights/yolov3-spp.weights", "cfg/yolov3-spp.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading image
cap = cv2.VideoCapture(0)

count=50
font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0
while True:
    _, frame = cap.read()
    frame_id += 1

    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    count=0
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.3)
    k = 0
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]]).lower()
            if label == 'person':
                if (count%50==0):
                    k = k + 1
                    confidence = confidences[i]
                    color = colors[class_ids[i]]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
                    cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 3, color, 2)
                    print ('Detected at Lat:17.443436 & Long:78.374243')
                    temp = k;
                    webApi();


    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 50), font, 4, (0, 0, 0), 3)
    print ('No. Persons Detected:',k)
    cv2.imshow("Image", frame)
    count = count+1
    key = cv2.waitKey(1)
    if key == 27:
        break
print ('Detected at Lat:17.443436 & Long:78.374243')
cap.release()
cv2.destroyAllWindows()