import sys
import csv
import depthai as dai #pylint: disable=import-error
import tensorflow as tf
import numpy as np
import cv2
from datetime import datetime
from pathlib import Path

from tensorflow.python.saved_model import tag_constants #pylint: disable=no-name-in-module

sys.path.insert(0, '../../')
from src.silhouette_segmenter import SilhouetteSegmenter

DEVICE_MX_ID = "14442C10B1BE60D700"
MYRIAD_DEVICE = "MYRIAD.1.2.3-ma2480"

class OnlinePipeline:
    def __init__(
        self,
        cnn_gait_recognition_model_path,
        mobilenet_ssd_model_path,
        segmentation_model_path
        ):
        # load classification model
        cnn_gait_recognition_model = tf.saved_model.load(cnn_gait_recognition_model_path, tags=[tag_constants.SERVING])
        # define the input signature
        infer = cnn_gait_recognition_model.signatures['serving_default']
        # test the model with a zero input
        infer(tf.reshape(tf.zeros([220, 220], tf.float32), [-1, 220, 220, 1]))
        # initialize the Mobilenet-SSD model path
        nn_path = str(Path(__file__).parent.resolve().absolute() / mobilenet_ssd_model_path)
        # check if the model exists
        if not Path(nn_path).exists():
            raise FileNotFoundError('Object detection model not found')
        # initialize the depthai pipeline
        pipeline = dai.Pipeline()
        # create a node for the Mobilenet-SSD model
        nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
        # define pipeline input and output
        xin_frame = pipeline.create(dai.node.XLinkIn)
        nn_out = pipeline.create(dai.node.XLinkOut)
        # set the stream names
        xin_frame.setStreamName("inFrame")
        nn_out.setStreamName("nn")
        # define the nn node properties
        nn.setConfidenceThreshold(0.5)
        nn.setBlobPath(nn_path)
        nn.setNumInferenceThreads(2)
        nn.input.setBlocking(False)
        # define the input and output stream links
        xin_frame.out.link(nn.input)
        nn.out.link(nn_out.input)
        # initialize the silhouette segmenter
        segmenter = SilhouetteSegmenter(model_path=segmentation_model_path, device=MYRIAD_DEVICE, binarization_th=0.4)
        # define OAK-D device
        found, device_info = dai.Device.getDeviceByMxId(DEVICE_MX_ID)
        if not found:
            raise RuntimeError("Device not found!")
        self.infer = infer
        self.pipeline = pipeline
        self.segmenter = segmenter
        self.device_info = device_info

    def get_binarized_silhouette(self, roi: np.ndarray) -> np.ndarray:
        sil_centered = np.zeros((220,220), np.uint8)
        # prepare ROI for segmentation
        roi_height = roi.shape[0]
        roi_width = roi.shape[1]
        resized_roi = cv2.resize(roi, (128,128))
        try:
            # run image segmentation on roi
            out = self.segmenter.sil_segmentation(resized_roi)
            # recover silhouette original size
            sil = cv2.resize(out, (roi_width, roi_height))
            if sil.mean() < 2:
                raise ValueError('Invalid silhouette')
            # center the binarized silhouette
            sil_centered = self.segmenter.sil_centering(sil, 220)
        except Exception as e: # pylint: disable=broad-except
            print(f'Found: {e}')
        return sil_centered

    def save_data_log(self, class_id, subject, walk, pred_acc, view):
        row = [datetime.now().strftime("%m-%d-%Y--%H-%M") + "," + str(class_id).zfill(3) + "," + subject + "," + walk + "," + str(pred_acc)]
        print(row)
        with open(f'data_log_{view}.csv', 'a', encoding='utf-8') as f:
            # create the csv writer
            writer = csv.writer(f)
            # write a row to the csv file
            writer.writerow(row)

    def get_classification_id(self, silhouettes):
        gei = np.zeros((220,220), np.uint8)
        try:
            # Compute the silhouettes array to obtain GEI
            gei = np.mean(np.array(silhouettes), axis=0).astype('uint8')
            # convert GEI to classification model input format
            gei_pred = gei/255.0
            gei_pred = tf.reshape(gei_pred, [-1, 220, 220, 1])
            gei_pred = tf.dtypes.cast(gei_pred, tf.float32)
            # run inference
            pred = np.asarray(self.infer(gei_pred)['output_1'])
            # get the class ID and prediction certainty
            class_id = int(tf.argmax(pred, axis = 1))
            pred_acc = round(pred[0][class_id], 2)
            #cv2.imshow("GEI", gei)
        except Exception as e: # pylint: disable=broad-except
            print(f'Found: {e}')
        return class_id, pred_acc, gei

    # nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
    def _frame_norm(self, frame, bbox):
        norm_vals = np.full(len(bbox), frame.shape[0])
        norm_vals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int)

    def to_planar(self, arr: np.ndarray, shape: tuple) -> np.ndarray:
        return cv2.resize(arr, shape).transpose(2, 0, 1).flatten()

    def get_roi(self, frame, detection):
        # call normalization function
        bbox = self._frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
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
