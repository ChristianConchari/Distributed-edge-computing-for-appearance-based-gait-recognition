#!/usr/bin/env python3
from utils import GEI_generation
from utils.myriad_segmentation import Infer_myriad
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
import argparse
import sys
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

parser = argparse.ArgumentParser()
parser.add_argument('--subject', type=str, required=True, help='Subject number from dataset OAKGait16')

args = parser.parse_args()

# set node position angle
view = '075'
# set edge-node name
jetson = 'jetson6'

# load classification model
cnn_gait_recognition = tf.saved_model.load(f"/home/{jetson}/models/cnn_gait_recognition_acc_09520_val_acc_0.8883_TFTRT_FP16", tags=[tag_constants.SERVING])
infer = cnn_gait_recognition.signatures['serving_default']

# finish the model loading
infer(tf.reshape(tf.zeros([160, 160], tf.float32), [-1, 160, 160, 1]))

# load object detection model
nnPathDefault = str((Path(__file__).parent / Path(f'/home/{jetson}/models/mobilenet-ssd_openvino_2021.4_6shave.blob')).resolve().absolute())

# raise exception if model does not exist
if not Path(nnPathDefault).exists():
    raise FileNotFoundError(f'Required file/s not found, please run "{sys.executable} install_requirements.py"')

# define MobilenetSSD label texts
labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]



# create pipeline
pipeline = dai.Pipeline()

# define pipeline for RGB camera capturing
camRgb = pipeline.create(dai.node.ColorCamera)
# create pipeline for object detection
nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
xoutRgb = pipeline.create(dai.node.XLinkOut)
nnOut = pipeline.create(dai.node.XLinkOut)

# define stream names
xoutRgb.setStreamName("rgb")
nnOut.setStreamName("nn")

# define image acquisition properties
camRgb.setPreviewSize(300, 300)
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setInterleaved(False)
camRgb.setFps(30)

# define neural network properties
nn.setConfidenceThreshold(0.5)
nn.setBlobPath(nnPathDefault)
nn.setNumInferenceThreads(2)
nn.input.setBlocking(False)

# linking Rgb with nn stream
nn.passthrough.link(xoutRgb.input)

camRgb.preview.link(nn.input)
nn.out.link(nnOut.input)

# initilize image segmentation script
seg = Infer_myriad()

# define empty variables
silhouettes = []
classID = None
GEI = np.zeros((200,200), np.uint8)
out = np.zeros((128,128), np.uint8)

# define the OAK-D device ID
found, device_info = dai.Device.getDeviceByMxId("14442C1041F042D700")

# Raise exception error if OAK-D device is not found
if not found:
    raise RuntimeError("Device not found!")

# define socket for communication
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect(("8.8.8.8", 80))
ip_address = str(s.getsockname()[0])
s.close()

# define port for communication
port = '5555'

# initialize communication through imagezmq 
sender = imagezmq.ImageSender(connect_to='tcp://'+ ip_address + ':' + port, REQ_REP=False)
jetson_name = socket.gethostname()
sender.zmq_socket.set_hwm(2)
sender.zmq_socket.setsockopt(imagezmq.zmq.SNDHWM,10)

# call timestamp
now = datetime.now()


def create_dir(folder):
    if not os.path.isdir(folder):
        print("Creating :" + folder)
        os.mkdir(folder)

# create directory if current day directory is not created
data_recordings = f"/home/{jetson}/data_recordings"
gei_path = os.path.join(data_recordings, now.strftime("%m_%d_%Y") + f"_{view}_GEI")
classification_path = os.path.join(data_recordings, now.strftime("%m_%d_%Y") + f"_{view}-classID")

create_dir(data_recordings)
create_dir(gei_path)
create_dir(classification_path)

# connect to device and start pipeline
with dai.Device(pipeline, device_info) as device:

    # output queues will be used to get the rgb frames and nn data from the outputs defined above
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    detections = []
    startTime = time.monotonic()
    counter = 0
    color = (0, 255, 0)
    jpeg_quality = 95
    in_counter = 0
    number_generated_GEIs = 0

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

    # create directories for storing data
    def create_dir(folder, force=True, verbose=False):
        try:
            os.makedirs(folder) 
        
        except:
            if force:
                shutil.rmtree(folder)
                os.makedirs(folder)

    save_frame_path = os.path.join(f"/home/{jetson}/test_frames", args.subject, view)
    create_dir(save_frame_path)
    save_GEI_path = os.path.join(f"/home/{jetson}/test_GEIs", args.subject, view)
    create_dir(save_GEI_path)

    while True:
        
        # get RGB data
        inRgb = qRgb.get()
        # get detection data
        inDet = qDet.get()

        counter += 1

        if inRgb is not None:
            frame = inRgb.getCvFrame()
            cv2.putText(frame, "NN fps: {:.2f}".format(counter / (time.monotonic() - startTime)),
                        (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 1, color)
        
        if inDet is not None:
            detections = inDet.detections
            
        # If the frame is available, draw bounding boxes on it and show the frame
        for detection in detections:
            
            # If person is detected
            if labelMap[detection.label] == 'person':

                # Return false detections counter to zero
                in_counter = 0

                # call normalization function
                bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))

                # apply offsets aiming to center the person
                x, y, w, h = bbox
                offsets = [10, 10, 10, 10]
                x = x - offsets[0] if x - offsets[0] >=0 else 0
                y = y - offsets[3] if y - offsets[3] >=0 else 0
                w = w + offsets[1]+offsets[0] if w + offsets[1]+offsets[0] <=300 else 300
                h = h + offsets[3]+offsets[1] if h + offsets[3]+offsets[1] <=300 else 300
                bbox = [x, y, w, h]

                # extract region of interest from bounding-box coordinates
                roi = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

                # convert roi to segmentation model input format
                roi = cv2.resize(roi,(128,128))
                roi = roi/255.0
                roi = tf.reshape(roi,[-1,128,128,3])

                # obtain segmented images
                out = seg.infer(roi)

                # append normalized silhouettes to list
                silhouettes.append(out)

                # compute gei for 30 silhouettes, approx. 1.08 seconds of walking at approx. 30 FPS
                if len(silhouettes) % 30 == 0:

                    # compute the silhouettes array to obtain GEI
                    GEI, _ = GEI_generation.GEI_generator(silhouettes)
                    GEI = np.array(GEI, dtype='uint8')
                    
                    # convert GEI to classification model input format
                    GEI_infer = GEI/255.0
                    GEI_pred = tf.reshape(GEI_infer, [-1, 160, 160, 1])
                    GEI_pred = tf.dtypes.cast(GEI_pred, tf.float32)

                    number_generated_GEIs += 1
                    
                    # run inference
                    pred = np.asarray(infer(GEI_pred)['output_1'])

                    # get the class ID and prediction certainty
                    classID = int(tf.argmax(pred, axis = 1))
                    pred_acc = round(pred[0][classID], 2)

                    # check if identified subject correspond to video subject
                    if str(classID).zfill(3) == subject:
                        correct_classification += 1
                        color = (0, 255, 0)
                    else:
                        color = (0, 0, 255)
                    
                    # save relevant information
                    row = [now.strftime("%m-%d-%Y--%H-%M") + "," + str(classID).zfill(3) + "," + args.subject + "," + str(pred_acc) + "," + str(round(time.monotonic() - startTime,2))]
                    print(row)

                    with  open(f'data_log_{view}.csv', 'a') as f:
                        
                        # create the csv writer
                        writer = csv.writer(f)
                        # write a row to the csv file
                        writer.writerow(row)

                # clean silhouettes array if 40 silhouettes is reached
                if (len(silhouettes) > 30):
                        silhouettes = []


                # if subject is not identified show wait message, else show subject ID                                    
                if classID is None:
                    text_id = "wait"
                else:
                    text_id = str(classID).zfill(3)


                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0]+80, bbox[1]-40), color, -1)                
                cv2.putText(frame, text_id, (bbox[0] + 5, bbox[1] - 10), cv2.FONT_HERSHEY_TRIPLEX, 1, (0,0,0), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

                if classID is not None:
                    cv2.imwrite(os.path.join(save_frame_path, f'{args.subject}-{str(classID).zfill(3)}-{view}-{now.strftime("%m-%d-%Y--%H-%M-%S")}-{number_generated_GEIs}.jpg'), frame)
                    cv2.imwrite(os.path.join(save_GEI_path, f'{args.subject}-{str(classID).zfill(3)}-{view}-{now.strftime("%m-%d-%Y--%H-%M-%S")}-{number_generated_GEIs}.jpg'), GEI)
                    

            else:
                in_counter += 1  
                if in_counter > 80:
                    silhouettes = []
                    classID = None
                    GEI = np.zeros((160,160), np.uint8)
                    out = np.zeros((128,128), np.uint8)
                    in_counter = 0

        GEI_resized = cv2.resize(GEI, (300, 300))
        out_resized = cv2.resize(out, (300, 300))

        try:
            out_resized = cv2.cvtColor(out_resized, cv2.COLOR_GRAY2RGB)
            GEI_resized = cv2.cvtColor(GEI_resized, cv2.COLOR_GRAY2RGB)
        except:
            pass

        # concat important images to send to edge-server
        cimg = vconcat_resize([frame, out_resized, GEI_resized])
        cimg = cv2.resize(cimg, (150, 450))
        
        

        # define JPG encode parameters
        ret_code, jpg_buffer = cv2.imencode(".jpg", cimg, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        # send image to edge-server
        sender.send_jpg(jetson_name, jpg_buffer)
        
        if cv2.waitKey(1) == ord('q'):
            break

    # desconect imagezmq sender
    sender.zmq_socket.disconnect('tcp://'+ ip_address+ ':' + port)