import sys
import os
import cv2
import depthai as dai #pylint: disable=import-error
from time import monotonic
sys.path.insert(0, '../../')
from src.create_dir import create_dir
from online_pipeline import OnlinePipeline

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

CNN_GAIT_RECOGNITION_MODEL_PATH = "../../models/cnn_gait_recognition_acc_0.9514_loss_0.3384_val_acc_0.9361_loss_acc_0.4261_TFTRT_FP16"
MOBILENET_SSD_MODEL_PATH = "../../models/mobilenet-ssd_openvino_2021.4_6shave.blob"
LABEL_MAP = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
SEGMENTATION_MODEL_PATH = "../../models/128x128_acc_0.8786_loss_0.1018_val-acc_0.8875_val-loss_0.0817_0.22M_13-09-22-TF_OAKGait16.xml"

TEST_FRAMES_RESULT_PATH = "../../test_frames"
TEST_GEIS_RESULT_PATH = "../../test_GEIs"
TEST_CLIPS = "../../test_clips"

def main():
    op = OnlinePipeline(
        cnn_gait_recognition_model_path=CNN_GAIT_RECOGNITION_MODEL_PATH,
        mobilenet_ssd_model_path=MOBILENET_SSD_MODEL_PATH,
        segmentation_model_path=SEGMENTATION_MODEL_PATH
    )
    # connect to device and start pipeline
    with dai.Device(op.pipeline, op.device_info) as device:
        # input queue will be used to send video frames to the device.
        q_in = device.getInputQueue(name="inFrame")
        # output queue will be used to get nn data from the video frames.
        q_det = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
        views = sorted(os.listdir(TEST_CLIPS))
        for view in views:
            subjects = sorted(os.listdir(os.path.join(TEST_CLIPS,view)))
            for subject in subjects:
                walks = sorted(os.listdir(os.path.join(TEST_CLIPS,view,subject)))
                save_frame_path = os.path.join(TEST_FRAMES_RESULT_PATH, subject, view)
                create_dir(save_frame_path)
                save_gei_path = os.path.join(TEST_GEIS_RESULT_PATH, subject, view)
                create_dir(save_gei_path)
                for walk in walks:
                    video_files = sorted(os.listdir(os.path.join(TEST_CLIPS,view,subject,walk)))
                    number_generated_geis = 0
                    for video_path in video_files:
                        detections = []
                        color = (0, 0, 255)
                        silhouettes = []
                        class_id = None
                        print(f"Processing: subject {subject} in view {view} in walk {walk}")
                        cap = cv2.VideoCapture(os.path.join(TEST_CLIPS,view,subject,walk,video_path))
                        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        print(length)
                        if length > 85:
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
                            img.setData(op.to_planar(frame, (300, 300)))
                            img.setWidth(300)
                            img.setHeight(300)
                            # send data to OAK-D device
                            q_in.send(img)
                            # get detections from object detection model
                            in_det = q_det.tryGet()
                            if in_det is not None:
                                detections = in_det.detections
                            if frame is not None:
                                for detection in detections:
                                    if LABEL_MAP[detection.label] == 'person':             
                                        # get the roi from the detection
                                        roi, bbox = op.get_roi(frame, detection)
                                        # get binarized silhouette
                                        sil_centered = op.get_binarized_silhouette(roi)
                                        #cv2.imshow(sil_centered)
                                        # append normalized silhouettes to list
                                        silhouettes.append(sil_centered)
                                        # compute gei for all the gathered silhouettes
                                        if len(silhouettes) % cfr == 0:
                                            number_generated_geis += 1
                                            class_id, pred_acc, gei = op.get_classification_id(silhouettes)
                                            # check if identified subject correspond to video subject
                                            color = (0, 255, 0) if str(class_id).zfill(3) == subject else (0, 0, 255)
                                            op.save_data_log(class_id, subject, walk, pred_acc, view)
                                        # clean silhouettes array if 40 silhouettes is reached
                                        if len(silhouettes) > cfr:
                                            silhouettes = []
                                        # if subject is not identified show wait message, else show subject ID                                    
                                        if class_id is None:
                                            text_id = "wait"
                                            color = (255, 0, 0)
                                        else:
                                            text_id = str(class_id).zfill(3)
                                        # draw bounding box and text
                                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0]+40, bbox[1]-20), color, -1)
                                        cv2.putText(frame, text_id, (bbox[0] + 2, bbox[1] - 5), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                                        # save identification frames
                                        if class_id is not None:
                                            cv2.imwrite(os.path.join(save_frame_path, f'{subject}-{str(class_id).zfill(3)}-{view}-{video_path.split(".")[0]}-{number_generated_geis}.jpg'), frame)
                                            cv2.imwrite(os.path.join(save_gei_path, f'{subject}-{str(class_id).zfill(3)}-{view}-{video_path.split(".")[0]}-{number_generated_geis}.jpg'), gei)
                                        #cv2.imshow("rgb", frame)
                            if cv2.waitKey(1) == ord('q'):
                                break
if __name__ == "__main__":
    main()
