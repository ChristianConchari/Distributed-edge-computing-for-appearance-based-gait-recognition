#!/usr/bin/env python3
from utils import GEI_generation
from utils.myriad_segmentation import Infer_myriad
from pathlib import Path
import sys
import cv2
import shutil
import depthai as dai
import numpy as np
import csv
from datetime import datetime
from time import monotonic
import time
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

# set node position angle
view = '075'
# set edge-node name
jetson = 'jetson6'

# load classification model
cnn_gait_recognition = tf.saved_model.load(f"/home/{jetson}/models/cnn_gait_recognition_acc_09520_val_acc_0.8883_TFTRT_FP16", tags=[tag_constants.SERVING])
infer = cnn_gait_recognition.signatures['serving_default']

# finish the model loading
infer(tf.reshape(tf.zeros([160, 160], tf.float32), [-1, 160, 160, 1]))

# Get argument first

parentDir = Path(__file__).parent
nnPath = str((parentDir / Path(f'/home/{jetson}/models/mobilenet-ssd_openvino_2021.4_6shave.blob')).resolve().absolute())

if len(sys.argv) > 2:
    nnPath = sys.argv[1]
    videoPath = sys.argv[2]

if not Path(nnPath).exists():
    import sys
    raise FileNotFoundError(f'Object detection model not found')

# MobilenetSSD label texts
labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


# create pipeline
pipeline = dai.Pipeline()

# create pipeline for object detection
nn = pipeline.create(dai.node.MobileNetDetectionNetwork)

# define pipeline input and outputs
xinFrame = pipeline.create(dai.node.XLinkIn)
nnOut = pipeline.create(dai.node.XLinkOut)

xinFrame.setStreamName("inFrame")
nnOut.setStreamName("nn")

# define neural network properties
nn.setConfidenceThreshold(0.5)
nn.setBlobPath(nnPath)
nn.setNumInferenceThreads(2)
nn.input.setBlocking(False)

# Define links
xinFrame.out.link(nn.input)
nn.out.link(nnOut.input)

# initialize segmentation script
seg = Infer_myriad()

# define OAK-D device
found, device_info = dai.Device.getDeviceByMxId("14442C1041F042D700")

if not found:
    raise RuntimeError("Device not found!")

# connect to device and start pipeline
with dai.Device(pipeline, device_info) as device:

    # input queue will be used to send video frames to the device.
    qIn = device.getInputQueue(name="inFrame")
    # output queue will be used to get nn data from the video frames.
    qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    # nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
    def frameNorm(frame, bbox):
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

    
    # Define test clips directory
    test_clips = f"/home/{jetson}/test_clips"
    views = sorted(os.listdir(test_clips))
    
    for view in views:
        subjects = sorted(os.listdir(os.path.join(test_clips,view)))

        for subject in subjects:
            walks = sorted(os.listdir(os.path.join(test_clips,view,subject)))
            save_frame_path = os.path.join(f"/home/{jetson}/test_frames", subject, view)
            create_dir(save_frame_path)
            save_GEI_path = os.path.join(f"/home/{jetson}/test_GEIs", subject, view)
            create_dir(save_GEI_path)
            
            
            for walk in walks:
                video_files = sorted(os.listdir(os.path.join(test_clips,view,subject,walk)))
                correct_classification = 0
                number_generated_GEIs = 0

                for videoPath in video_files:
                    detections = []
                    now = datetime.now()
                    color = (0, 0, 255)
                    silhouettes = []
                    classID = None
                    out = np.zeros((128,128), np.uint8)
                    sil_aligned = np.zeros((200,200), np.uint8)
                    GEI = np.zeros((200,200), np.uint8)
                    
                    print(f"Processing: subject {subject} in view {view} in walk {walk}")
                    cap = cv2.VideoCapture(os.path.join(test_clips,view,subject,walk,videoPath))
                    
                    
                    while cap.isOpened():
                        read_correctly, frame = cap.read()
                        
                        if not read_correctly:
                            break

                        # convert frame to detection model input format
                        frame = cv2.resize(frame,(300,300), interpolation=cv2.INTER_LINEAR)
                        scaled_frame = frame.transpose(2, 0, 1).flatten()
                        
                        # define image object for depthai processing
                        img = dai.ImgFrame()
                        img.setTimestamp(monotonic())
                        img.setData(scaled_frame)
                        img.setWidth(300)
                        img.setHeight(300)

                        # send data to OAK-D device
                        qIn.send(img)
                        # get detections from object detection model
                        inDet = qDet.tryGet()

                        if inDet is not None:
                            detections = inDet.detections
                            

                        if frame is not None:
                            
                            for detection in detections:
                                if labelMap[detection.label] == 'person':             
                                    bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                                    x, y, w, h = bbox
                                    
                                    # apply offsets aiming to center the person
                                    offsets = [10, 10, 10, 10]
                                    x = x - offsets[0] if x - offsets[0] >=0 else 0
                                    y = y - offsets[3] if y - offsets[3] >=0 else 0
                                    w = w + offsets[1]+offsets[0] if w + offsets[1]+offsets[0] <=300 else 300
                                    h = h + offsets[3]+offsets[1] if h + offsets[3]+offsets[1] <=300 else 300
                                    bbox = [x, y, w, h]
                                    
                                    # extract region of interest from bounding-box coordinates
                                    roi = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                                    
                                    if roi.shape[1] > 60  or roi.shape[1] < 150:
                                        # convert roi to segmentation model input format
                                        roi = cv2.resize(roi,(128,128))
                                
                                        # Run image segmentation on roi
                                        out = seg.infer(roi)
                                        #cv2.imshow("silhouette", out)
                                        # append normalized silhouettes to list
                                        silhouettes.append(out)

                                    # compute gei for 40 silhouettes, approx. 1.33 seconds of walking at approx. 30 FPS
                                    if len(silhouettes) % 30 == 0:
                                        try:
                                            # Compute the silhouettes array to obtain GEI
                                            GEI, _ = GEI_generation.GEI_generator(silhouettes)

                                            if np.mean(GEI) < 30: 
                                                classID = None
                                                raise
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

                                            #cv2.imshow("GEI", np.array(GEI, dtype='uint8'))


                                            # save relevant information
                                            row = [now.strftime("%m-%d-%Y--%H-%M") + "," + str(classID).zfill(3) + "," + subject + "," + walk + "," + str(pred_acc) ]
                                            print(row)

                                            with  open(f'data_log_{view}.csv', 'a') as f:
                                                
                                                # create the csv writer
                                                writer = csv.writer(f)
                                                # write a row to the csv file
                                                writer.writerow(row)
                                            
                                        except:
                                            pass

                                    # clean silhouettes array if 40 silhouettes is reached
                                    if len(silhouettes) > 30:
                                        silhouettes = []

                                    
                                    # if subject is not identified show wait message, else show subject ID                                    
                                    if classID is None:
                                        text_id = "wait"
                                        color = (255, 0, 0)
                                    else:
                                        text_id = str(classID).zfill(3)
                                    
                                    
                                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0]+40, bbox[1]-20), color, -1)
                                    cv2.putText(frame, text_id, (bbox[0] + 2, bbox[1] - 5), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                                    
                                    # save identification frames
                                    if classID is not None:
                                        cv2.imwrite(os.path.join(save_frame_path, f'{subject}-{str(classID).zfill(3)}-{view}-{videoPath.split(".")[0]}-{number_generated_GEIs}.jpg'), frame)
                                        cv2.imwrite(os.path.join(save_GEI_path, f'{subject}-{str(classID).zfill(3)}-{view}-{videoPath.split(".")[0]}-{number_generated_GEIs}.jpg'), GEI)
                                        

                                    #cv2.imshow("rgb", frame)
                                    #cv2.imshow("silhouette", out)

 
                        if cv2.waitKey(1) == ord('q'):
                            break