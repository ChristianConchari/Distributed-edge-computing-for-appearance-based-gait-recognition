import sys
from pathlib import Path
import cv2
import depthai as dai #pylint: disable=import-error
import tensorflow as tf #pylint: disable=import-error
import numpy as np
from tensorflow.python.saved_model import tag_constants #pylint: disable=no-name-in-module, import-error
import socket
import imagezmq

sys.path.insert(0, '../../')
from src.silhouette_segmenter import SilhouetteSegmenter

DEVICE_MX_ID = "14442C10B1BE60D700"
MYRIAD_DEVICE = "MYRIAD.1.2.3-ma2480"
PORT = '5555'

class OnlinePipeline:
    def __init__(
        self,
        cnn_gait_recognition_model_path,
        mobilenet_ssd_model_path,
        segmentation_model_path,
        is_video=False
        ):
        # define the port for communication
        self.port = PORT
        # load classification model
        cnn_gait_recognition_model = tf.saved_model.load(cnn_gait_recognition_model_path, tags=[tag_constants.SERVING])
        # define the input signature
        self.infer = cnn_gait_recognition_model.signatures['serving_default']
        # test the model with a zero input
        self.infer(tf.reshape(tf.zeros([220, 220], tf.float32), [-1, 220, 220, 1]))
        # initialize the Mobilenet-SSD model path
        nn_path = str(Path(__file__).parent.resolve().absolute() / mobilenet_ssd_model_path)
        # check if the model exists
        if not Path(nn_path).exists():
            raise FileNotFoundError('Object detection model not found')
        # initialize the dai pipeline
        if not is_video:
            self.pipeline = self._initialize_dai_pipeline(nn_path)
        else:
            self.pipeline = self._initialize_dai_pipeline_video(nn_path)
        # initialize the silhouette segmenter
        self.segmenter = SilhouetteSegmenter(model_path=segmentation_model_path, device=MYRIAD_DEVICE, binarization_th=0.3)
        # initialize the imagezmq sender
        if not is_video:
            self.sender, self.jetson_name, self.ip_address = self._initialize_imagezmq()
        # define OAK-D device
        found, self.device_info = dai.Device.getDeviceByMxId(DEVICE_MX_ID)
        if not found:
            raise RuntimeError("Device not found!")
        
    def _initialize_imagezmq(self):
        # define socket for communication
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            # Get the local IP address
            ip_address = str(s.getsockname()[0])

        # initialize communication through imagezmq
        sender = imagezmq.ImageSender(connect_to='tcp://'+ ip_address + ':' + PORT, REQ_REP=False)
        jetson_name = socket.gethostname()
        sender.zmq_socket.set_hwm(2)
        sender.zmq_socket.setsockopt(imagezmq.zmq.SNDHWM,10)
        return sender, jetson_name, ip_address
        
    def _initialize_dai_pipeline(self, nn_path):
        # create pipeline
        pipeline = dai.Pipeline()
        # define pipeline for RGB camera capturing
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        # create pipeline for object detection
        nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
        xout_rgb = pipeline.create(dai.node.XLinkOut)
        nn_out = pipeline.create(dai.node.XLinkOut)
        # define stream names
        xout_rgb.setStreamName("rgb")
        nn_out.setStreamName("nn")
        # define image acquisition properties
        cam_rgb.setPreviewSize(300, 300)
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
        cam_rgb.setInterleaved(False)
        cam_rgb.setFps(30)
        # define neural network properties
        nn.setConfidenceThreshold(0.5)
        nn.setBlobPath(nn_path)
        nn.setNumInferenceThreads(2)
        nn.input.setBlocking(False)
        # linking Rgb with nn stream
        nn.passthrough.link(xout_rgb.input)
        # linking Rgb with nn input
        cam_rgb.preview.link(nn.input)
        nn.out.link(nn_out.input)
        return pipeline
    
    def _initialize_dai_pipeline_video(self, nn_path):
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
        return pipeline

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
            pass
        return sil_centered, out

    def get_classification_id(self, silhouettes):
        gei = np.zeros((220,220), np.uint8)
        class_id = None
        try:
            # Compute the silhouettes array to obtain GEI
            gei = np.mean(np.array(silhouettes), axis=0).astype('uint8')
            # convert GEI to classification model input format
            gei_pred = gei/255.0
            gei_pred = tf.reshape(gei_pred, [-1, 220, 220, 1])
            gei_pred = tf.dtypes.cast(gei_pred, tf.float32)
            # run inference
            pred = np.asarray(self.infer(gei_pred)['dense_1'])
            # get the class ID and prediction certainty
            class_id = int(tf.argmax(pred, axis = 1))
            class_id = class_id if class_id < 6 else class_id + 1
            #cv2.imshow("GEI", gei)
        except Exception as e: # pylint: disable=broad-except
            pass
        return class_id, gei

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

    def vconcat_resize(self, img_list, interpolation=cv2.INTER_CUBIC):
        # Find the minimum width among the images
        w_min = min(img.shape[1] for img in img_list)
        # Resize images to have the same width while maintaining aspect ratio
        im_list_resize = [
            cv2.resize(img, (w_min, int(img.shape[0] * w_min / img.shape[1])), interpolation=interpolation)
            for img in img_list
        ]
        # Concatenate the resized images vertically
        return cv2.vconcat(im_list_resize)
