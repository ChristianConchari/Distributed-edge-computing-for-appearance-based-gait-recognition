import sys
import os
sys.path.insert(0, '../../')

from sklearn.metrics import classification_report
from src.create_dir import create_dir
from src.silhouette_segmenter import SilhouetteSegmenter
from utils import GEI_generation
from utils.myriad_segmentation import Infer_myriad
from pathlib import Path
from datetime import datetime
from time import monotonic
from tensorflow.python.saved_model import tag_constants

import sys
import cv2
import shutil
import depthai as dai
import numpy as np
import shutil
import csv
import time
import tensorflow as tf

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
CNN_GAIT_RECOGNITION_MODEL_PATH = "../../models/cnn_gait_recognition_acc_0.9514_loss_0.3384_val_acc_0.9361_loss_acc_0.4261_TFTRT_FP16"
MOBILENET_SSD_MODEL_PATH = "../../models/mobilenet-ssd_openvino_2021.4_6shave.blob"
LABEL_MAP = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
MYRIAD_DEVICE = "MYRIAD.1.2.3-ma2480"
SEGMENTATION_MODEL_PATH = "../../models/128x128_acc_0.8786_loss_0.1018_val-acc_0.8875_val-loss_0.0817_0.22M_13-09-22-TF_OAKGait16.xml"
DEVICE_MX_ID = "14442C10B1BE60D700"
TEST_FRAMES_RESULT_PATH = "../../test_frames"
TEST_GEIS_RESULT_PATH = "../../test_GEIs"
TEST_CLIPS = "../../test_clips"

def _get_binarized_silhouette(segmenter: SilhouetteSegmenter, roi: np.ndarray) -> np.ndarray: 
    sil_centered = np.zeros((220,220), np.uint8)
    # prepare ROI for segmentation     
    roi_height = roi.shape[0]
    roi_width = roi.shape[1]
    resized_roi = cv2.resize(roi, (128,128))         
    try:
        # run image segmentation on roi
        out = segmenter.sil_segmentation(resized_roi)
        # recover silhouette original size
        sil = cv2.resize(out, (roi_width, roi_height))
        if sil.mean() < 2:
            raise ValueError('Invalid silhouette')
        # center the binarized silhouette
        sil_centered = segmenter.sil_centering(sil, 220)
    except Exception as e: # pylint: disable=broad-except
        print(f'Found: {e}')
    return sil_centered

def _save_data_log(classID, subject, walk, pred_acc, view):
    row = [datetime.now().strftime("%m-%d-%Y--%H-%M") + "," + str(classID).zfill(3) + "," + subject + "," + walk + "," + str(pred_acc) ]
    print(row)
    with  open(f'data_log_{view}.csv', 'a') as f:
        # create the csv writer
        writer = csv.writer(f)
        # write a row to the csv file
        writer.writerow(row)

def _get_classification_id(silhouettes, infer):
    gei = np.zeros((220,220), np.uint8)
    try:
        # Compute the silhouettes array to obtain GEI
        gei = np.mean(np.array(silhouettes), axis=0).astype('uint8')
        # convert GEI to classification model input format
        gei_pred = gei/255.0
        gei_pred = tf.reshape(gei_pred, [-1, 220, 220, 1])
        gei_pred = tf.dtypes.cast(gei_pred, tf.float32)
        # run inference
        pred = np.asarray(infer(gei_pred)['output_1'])
        # get the class ID and prediction certainty
        class_id = int(tf.argmax(pred, axis = 1))
        pred_acc = round(pred[0][class_id], 2)
        #cv2.imshow("GEI", gei)
    except Exception as e: # pylint: disable=broad-except
        print(f'Found: {e}')
    return class_id, pred_acc, gei

# nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
def _frameNorm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

def _to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
    return cv2.resize(arr, shape).transpose(2, 0, 1).flatten()

def _get_roi(frame, detection):
    # call normalization function
    bbox = _frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
    # apply offsets aiming to center the person
    x, y, w, h = bbox
    offsets = [10, 10, 10, 10]
    x = x - offsets[0] if x - offsets[0] >=0 else 0
    y = y - offsets[3] if y - offsets[3] >=0 else 0
    w = w + offsets[1]+offsets[0] if w + offsets[1]+offsets[0] <= frame.shape[1] else frame.shape[1]
    h = h + offsets[3]+offsets[1] if h + offsets[3]+offsets[1] <= frame.shape[0] else frame.shape[0]
    bbox = [x, y, w, h]
    # extract region of interest from bounding-box coordinates
    roi = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    return roi, bbox
        
def main():
    # load classification model
    cnn_gait_recognition_model = tf.saved_model.load(CNN_GAIT_RECOGNITION_MODEL_PATH, tags=[tag_constants.SERVING])
    # define the input signature
    infer = cnn_gait_recognition_model.signatures['serving_default']
    # test the model with a zero input
    infer(tf.reshape(tf.zeros([220, 220], tf.float32), [-1, 220, 220, 1]))
    # initialize the Mobilenet-SSD model path
    nn_path = str(Path(__file__).parent.resolve().absolute() / MOBILENET_SSD_MODEL_PATH)
    # check if the model exists
    if not Path(nn_path).exists():
        raise FileNotFoundError('Object detection model not found')
    # initialize the depthai pipeline
    pipeline = dai.Pipeline()
    # create a node for the Mobilenet-SSD model
    nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
    # define pipeline input and output
    xinFrame = pipeline.create(dai.node.XLinkIn)
    nnOut = pipeline.create(dai.node.XLinkOut)
    # set the stream names
    xinFrame.setStreamName("inFrame")
    nnOut.setStreamName("nn")
    # define the nn node properties
    nn.setConfidenceThreshold(0.5)
    nn.setBlobPath(nn_path)
    nn.setNumInferenceThreads(2)
    nn.input.setBlocking(False)
    # define the input and output stream links
    xinFrame.out.link(nn.input)
    nn.out.link(nnOut.input)
    # initialize the silhouette segmenter
    segmenter = SilhouetteSegmenter(model_path=SEGMENTATION_MODEL_PATH, device=MYRIAD_DEVICE, binarization_th=0.4)
    # define OAK-D device
    found, device_info = dai.Device.getDeviceByMxId(DEVICE_MX_ID)
    if not found:
        raise RuntimeError("Device not found!")

    # connect to device and start pipeline
    with dai.Device(pipeline, device_info) as device:
        # input queue will be used to send video frames to the device.
        qIn = device.getInputQueue(name="inFrame")
        # output queue will be used to get nn data from the video frames.
        qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
        views = sorted(os.listdir(TEST_CLIPS))
        for view in ['090']:
            subjects = sorted(os.listdir(os.path.join(TEST_CLIPS,view)))
            for subject in ['011']:
                walks = sorted(os.listdir(os.path.join(TEST_CLIPS,view,subject)))
                save_frame_path = os.path.join(TEST_FRAMES_RESULT_PATH, subject, view)
                create_dir(save_frame_path)
                save_GEI_path = os.path.join(TEST_GEIS_RESULT_PATH, subject, view)
                create_dir(save_GEI_path)
                for walk in walks:
                    video_files = sorted(os.listdir(os.path.join(TEST_CLIPS,view,subject,walk)))
                    number_generated_GEIs = 0
                    for videoPath in video_files:
                        detections = []
                        color = (0, 0, 255)
                        silhouettes = []
                        classID = None                        
                        print(f"Processing: subject {subject} in view {view} in walk {walk}")
                        cap = cv2.VideoCapture(os.path.join(TEST_CLIPS,view,subject,walk,videoPath))
                        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        print(length)
                        if length > 85 : 
                            cfr = 85
                        elif subject == '005':
                            cfr = 70
                        else:
                            cfr = int(length*0.85)
                        
                        while cap.isOpened():                            
                            read_correctly, frame = cap.read()
                            if not read_correctly:
                                break                            
                            # define image object for depthai processing
                            img = dai.ImgFrame()
                            img.setTimestamp(monotonic())
                            img.setData(_to_planar(frame, (300, 300)))
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
                                    if LABEL_MAP[detection.label] == 'person':             
                                        # get the roi from the detection
                                        roi, bbox = _get_roi(frame, detection)
                                        # get binarized silhouette
                                        sil_centered = _get_binarized_silhouette(segmenter, roi)
                                        #cv2.imshow(sil_centered)
                                        # append normalized silhouettes to list
                                        silhouettes.append(sil_centered)
                                        # compute gei for all the gathered silhouettes
                                        if len(silhouettes) % cfr == 0:
                                            number_generated_GEIs += 1
                                            classID, pred_acc, gei = _get_classification_id(silhouettes, infer)
                                            # check if identified subject correspond to video subject
                                            color = (0, 255, 0) if str(classID).zfill(3) == subject else (0, 0, 255)
                                            _save_data_log(classID, subject, walk, pred_acc, view)
                                        # clean silhouettes array if 40 silhouettes is reached
                                        if len(silhouettes) > cfr:
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
                                            cv2.imwrite(os.path.join(save_GEI_path, f'{subject}-{str(classID).zfill(3)}-{view}-{videoPath.split(".")[0]}-{number_generated_GEIs}.jpg'), gei)

                                        #cv2.imshow("rgb", frame)
                                        
                            if cv2.waitKey(1) == ord('q'):
                                break
if __name__ == "__main__":
    main()