#!/usr/bin/env python3
from utils import GEI_generation
from myriad_segmentation import infer_myriad
from pathlib import Path
import cv2
import csv
import depthai as dai
import numpy as np
from datetime import datetime
import time
import socket
import imagezmq
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
import os
import sys
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

# Load classification model
cnn_gait_recognition = tf.saved_model.load("cnn_gait_recognitionV6_TFTRT_FP16", tags=[tag_constants.SERVING])
infer = cnn_gait_recognition.signatures['serving_default']

# To finish the model loading
infer(tf.reshape(tf.zeros([160, 160], tf.float32), [-1, 160, 160, 1]))

# Load object detection model
nnPathDefault = str((Path(__file__).parent / Path('mobilenet-ssd_openvino_2021.4_6shave.blob')).resolve().absolute())

# Raise exception if model does not exist
if not Path(nnPathDefault).exists():
    raise FileNotFoundError(f'Required file/s not found, please run "{sys.executable} install_requirements.py"')

# Define MobilenetSSD label texts
labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
xoutRgb = pipeline.create(dai.node.XLinkOut)
nnOut = pipeline.create(dai.node.XLinkOut)

# Define stream names
xoutRgb.setStreamName("rgb")
nnOut.setStreamName("nn")

# Define image acquisition properties
camRgb.setPreviewSize(300, 300)
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setInterleaved(False)
camRgb.setFps(40)

# Define a neural network that will make predictions based on the source frames
nn.setConfidenceThreshold(0.5)
nn.setBlobPath(nnPathDefault)
nn.setNumInferenceThreads(2)
nn.input.setBlocking(False)

# Linking Rgb with nn stream
nn.passthrough.link(xoutRgb.input)

camRgb.preview.link(nn.input)
nn.out.link(nnOut.input)

# Initilize image segmentation script
seg = infer_myriad()

# Defines empty variables
silhouettes = []
classID = None
GEI = np.zeros((64,64), np.uint8)
out = np.zeros((128,128), np.uint8)

# Declares the OAK-D device ID
found, device_info = dai.Device.getDeviceByMxId("14442C1041F042D700")

# Raise exception error if OAK-D device is not found
if not found:
    raise RuntimeError("Device not found!")

# Get Wlan ip address
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect(("8.8.8.8", 80))
ip_address = str(s.getsockname()[0])
s.close()

# Set port for communication
port = '5555'

# Initialize communication through imagezmq 
sender = imagezmq.ImageSender(connect_to='tcp://'+ ip_address + ':' + port, REQ_REP=False)
jetson_name = socket.gethostname()
sender.zmq_socket.set_hwm(2)
sender.zmq_socket.setsockopt(imagezmq.zmq.SNDHWM,10)

# Call timestamp
dateTimeObj = datetime.now()

# Create directory if current day directory is not created
gei_path = os.path.join("data_recordings", str(dateTimeObj.year) + "-" + str(dateTimeObj.month) + "-" + str(dateTimeObj.day) + "075-GEI")

if not os.path.isdir(gei_path):
    print("Creating :" + gei_path)
    os.mkdir(gei_path)

# Create directory if current day directory is not created
classification_path = os.path.join("data_recordings", str(dateTimeObj.year) + "-" + str(dateTimeObj.month) + "-" + str(dateTimeObj.day) + "075-classID")

if not os.path.isdir(classification_path):
    print("Creating :" + classification_path)
    os.mkdir(classification_path)


# Connect to device and start pipeline
with dai.Device(pipeline, device_info) as device:

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    frame = None
    detections = []
    startTime = time.monotonic()
    counter = 0
    color2 = (0, 255, 0)
    color = (0, 255, 0)
    jpeg_quality = 95
    num = 0 
    in_counter = 0

    def vconcat_resize(img_list, interpolation = cv2.INTER_CUBIC):
        
        # Take minimum width and resize images
        w_min = min(img.shape[1] for img in img_list)
        im_list_resize = [
                         cv2.resize(img,(w_min, int(img.shape[0] * w_min / img.shape[1])), interpolation = interpolation)
                         for img in img_list
                         ]
        return cv2.vconcat(im_list_resize)    

    
    def frameNorm(frame, bbox):

        # nn data (bounding box locations in <0..1> range) normalization according to width/height input frame 
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)


    while True:

        inRgb = qRgb.get()
        inDet = qDet.get()

        if inRgb is not None:
            frame = inRgb.getCvFrame()
            cv2.putText(frame, "NN fps: {:.2f}".format(counter / (time.monotonic() - startTime)),
                        (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 1, color2)
        
        if inDet is not None:
            detections = inDet.detections
            counter += 1

        # If the frame is available, draw bounding boxes on it and show the frame
        for detection in detections:
            
            # If person is detected
            if labelMap[detection.label] == 'person':

                # Returns false detections counter to zero
                in_counter = 0

                # Calls normalization function
                bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))

                # Defines bounding box
                x, y, w, h = bbox
                offsets = [10, 10, 10, 10]
                x = x - offsets[0] if x - offsets[0] >=0 else 0
                y = y - offsets[3] if y - offsets[3] >=0 else 0
                w = w + offsets[1]+offsets[0] if w + offsets[1]+offsets[0] <=300 else 300
                h = h + offsets[3]+offsets[1] if h + offsets[3]+offsets[1] <=300 else 300
                bbox = [x, y, w, h]

                # Extract region of interest based on bounding box
                roi = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                roi = cv2.resize(roi,(128,128))
                roi = roi/255.0
                roi = tf.reshape(roi,[-1,128,128,3])

                # Run image segmentation on roi
                out = seg.infer(roi)

                # Append segmented silhouettes to list
                silhouettes.append(out)


                if len(silhouettes) % 40 == 0:

                    try:
                        # Run GEI generator script and resize to classification model input size
                        GEI, _ = GEI_generation.GEI_generator(silhouettes, size = 200, debug=False)
                        GEI_infer = GEI/255.0
                        GEI_pred = tf.reshape(GEI_infer, [-1, 160, 160, 1])
                        GEI_pred = tf.dtypes.cast(GEI_pred, tf.float32)
                        
                        # Run inference
                        pred = infer(GEI_pred)

                        # Get class ID according to dataset
                        classID = int(tf.argmax(np.asarray(pred['output_1']),axis=1))+1
                        
                        if classID == 6:
                            classID = 7
                        
                        row = [str(datetime.now()) + "," + "00"+str(classID) + "," + str(round(counter/(time.monotonic() - startTime),2)) + " FPS"]
                        print(row)

                        if not os.path.exists(f'data_log.csv'):
                            os.system("touch data_log.csv")

                        with  open('data_log.csv', 'a') as f:
                            # create the csv writer
                            writer = csv.writer(f)
                            # write a row to the csv file
                            writer.writerow(row)

                    
                    except Exception as e:
                        print("Error at GEI: ", e)

                    if (len(silhouettes) > 80):
                         silhouettes = []


                # Draw classification information in the image
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0]+80, bbox[1]-40), color, -1)
                
                if classID is None:
                    text_id = "wait"
                else:
                    text_id = "00"+str(classID)
                
                cv2.putText(frame, text_id, (bbox[0] + 5, bbox[1] - 10), cv2.FONT_HERSHEY_TRIPLEX, 1, (0,0,0), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)


            else:
                in_counter += 1  
                if in_counter > 80:
                    silhouettes = []
                    classID = None
                    GEI = np.zeros((64,64), np.uint8)
                    out = np.zeros((128,128), np.uint8)
                    in_counter = 0

        GEI_resized = np.array(GEI, dtype='uint8')  
        GEI_resized = cv2.resize(GEI_resized, (300, 300))
        out_resized = cv2.resize(out, (300, 300))
        try:
            out_resized = cv2.cvtColor(out_resized, cv2.COLOR_GRAY2RGB)
            GEI_resized = cv2.cvtColor(GEI_resized, cv2.COLOR_GRAY2RGB)
        except:
            pass

        cimg = vconcat_resize([frame, out_resized, GEI_resized])
        cimg = cv2.resize(cimg, (150, 450))
        
        if classID is not None:
            # Save GEI image in above defined path
            save_gei_path = os.path.join(gei_path, str(datetime.now())+str(classID))
            cv2.imwrite(f'{save_gei_path}.png', GEI)
            save_classification_path = os.path.join(classification_path, str(datetime.now())+'-'+"00"+str(classID))
            cv2.imwrite(f'{save_classification_path}.png',frame)

        ret_code, jpg_buffer = cv2.imencode(".jpg", cimg, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        sender.send_jpg(jetson_name, jpg_buffer)
        
        if cv2.waitKey(1) == ord('q'):
            break

    #client.disconnect()
    sender.zmq_socket.disconnect('tcp://'+ ip_address+ ':' + port)