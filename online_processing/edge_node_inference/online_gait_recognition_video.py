import sys
import os
import cv2
import depthai as dai #pylint: disable=import-error
from time import monotonic
sys.path.insert(0, '../../')
from src.create_dir import create_dir
from online_pipeline import OnlinePipeline
from datetime import datetime
import csv

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

CNN_GAIT_RECOGNITION_MODEL_PATH = "../../models/tftrt_cnn_gait_recognition_acc_0.9499_loss_0.3443_val_acc_0.8934_loss_acc_0.5382"
MOBILENET_SSD_MODEL_PATH = "../../models/mobilenet-ssd_openvino_2021.4_6shave.blob"
LABEL_MAP = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
SEGMENTATION_MODEL_PATH = "../../models/128x128_acc_0.8433_loss_0.0590_val-acc_0.8462_val-loss_0.0771_0.22M_13-09-22-TF_OAKGait16/model.xml"

TEST_FRAMES_RESULT_PATH = "../../test_frames"
TEST_GEIS_RESULT_PATH = "../../test_GEIs"
TEST_CLIPS = "../../test_clips"
CSV_DATA_PATH = "../../csv_data_video"

def main():
    op = OnlinePipeline(
        cnn_gait_recognition_model_path=CNN_GAIT_RECOGNITION_MODEL_PATH,
        mobilenet_ssd_model_path=MOBILENET_SSD_MODEL_PATH,
        segmentation_model_path=SEGMENTATION_MODEL_PATH,
        is_video=True
    )
    # connect to device and start pipeline
    with dai.Device(op.pipeline, op.device_info) as device:
        # input queue will be used to send video frames to the device.
        q_in = device.getInputQueue(name="inFrame")
        # output queue will be used to get nn data from the video frames.
        q_det = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
        views = sorted(os.listdir(TEST_CLIPS))
        for view in views:
            if not os.path.exists(CSV_DATA_PATH):
                create_dir(CSV_DATA_PATH)
            subjects = sorted(os.listdir(os.path.join(TEST_CLIPS,view)))
            with open(f'{CSV_DATA_PATH}/data_log_{view}.csv', 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["time_stamp", "pred", "label", "walk", "ssd_mobile_net_time", "segmentation_time", "cnn_time"])
            with open(f'{CSV_DATA_PATH}/fps_data_log_{view}.csv', 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["time_stamp", "fps"])
            for subject in subjects:
                walks = sorted(os.listdir(os.path.join(TEST_CLIPS,view,subject)))
                save_frame_path = os.path.join(TEST_FRAMES_RESULT_PATH, subject, view)
                create_dir(save_frame_path)
                save_gei_path = os.path.join(TEST_GEIS_RESULT_PATH, subject, view)
                create_dir(save_gei_path)
                past_log_time = monotonic()
                for walk in walks:
                    video_files = sorted(os.listdir(os.path.join(TEST_CLIPS,view,subject,walk)))
                    for video_path in video_files:
                        detections = []
                        color = (0, 0, 255)
                        silhouettes = []
                        segmentation_inference_times = []
                        mobilenet_inference_times = []
                        class_id = None
                        number_generated_geis = 0
                        print(f"Processing: subject {subject} in view {view} in walk {walk}")
                        cap = cv2.VideoCapture(os.path.join(TEST_CLIPS,view,subject,walk,video_path))

                        while cap.isOpened():
                            frame_start_time = monotonic()
                            read_correctly, frame = cap.read()
                            if not read_correctly:
                                break
                            # define image object for depthai processing
                            start_time = monotonic()
                            img = dai.ImgFrame()
                            img.setTimestamp(monotonic())
                            img.setData(op.to_planar(frame, (300, 300)))
                            img.setWidth(300)
                            img.setHeight(300)
                            # send data to OAK-D device
                            q_in.send(img)
                            # get detections from object detection model
                            in_det = q_det.tryGet()
                            end_time = monotonic()
                            mobilenet_inference_times.append(end_time - start_time)
                            if in_det is not None:
                                detections = in_det.detections
                            if frame is not None:
                                for detection in detections:
                                    if LABEL_MAP[detection.label] == 'person':             
                                        # get the roi from the detection
                                        roi, bbox = op.get_roi(frame, detection)
                                        # get binarized silhouette
                                        start_time = monotonic()
                                        sil_centered = op.get_binarized_silhouette(roi)
                                        end_time = monotonic()
                                        segmentation_inference_times.append(end_time - start_time)
                                        #cv2.imshow(sil_centered)
                                        # append normalized silhouettes to list
                                        silhouettes.append(sil_centered)
                                        # compute gei for 40 silhouettes
                                        if len(silhouettes) % 40 == 0 and len(silhouettes) != 0:
                                            data_silhouettes = silhouettes.copy()
                                            silhouettes = []
                                            number_generated_geis += 1
                                            start_time = monotonic()
                                            class_id, gei = op.get_classification_id(data_silhouettes)
                                            end_time = monotonic()
                                            cnn_inference_time = round(end_time - start_time, 4)
                                            row = [
                                                datetime.now().strftime("%m-%d-%Y--%H-%M"),
                                                str(class_id).zfill(3),
                                                subject,
                                                f'{walk}-{str(video_path.split(".")[0])}-{str(number_generated_geis)}',
                                                round(sum(mobilenet_inference_times) / len(mobilenet_inference_times), 4),
                                                round(sum(segmentation_inference_times) / len(segmentation_inference_times), 4),
                                                cnn_inference_time
                                            ]
                                            print(row)
                                            with open(f'../../csv_data/data_log_{view}.csv', 'a', encoding='utf-8', newline='') as f:
                                                writer = csv.writer(f)
                                                writer.writerow(row)
                                            # check if identified subject correspond to video subject
                                            color = (0, 255, 0) if str(class_id).zfill(3) == subject else (0, 0, 255)
                                        # if subject is not identified show wait message, else show subject ID                                    
                                        if class_id is None:
                                            text_id = "wait"
                                            color = (255, 0, 0)
                                        else:
                                            text_id = str(class_id).zfill(3)
                                        
                                        frame_end_time = monotonic()
                                        fps = round(1 / (frame_end_time - frame_start_time), 2)
                                        if monotonic() - past_log_time > 10:
                                            past_log_time = monotonic()
                                            with open(f'../../csv_data/fps_data_log_{view}.csv', 'a', encoding='utf-8', newline='') as f:
                                                writer = csv.writer(f)
                                                writer.writerow([datetime.now().strftime("%m-%d-%Y--%H-%M"), fps])
                                        # draw bounding box, text and fps
                                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0]+40, bbox[1]-20), color, -1)
                                        cv2.putText(frame, text_id, (bbox[0] + 2, bbox[1] - 5), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                                        # save identification frames
                                        if class_id is not None:
                                            tmp_class_id = class_id
                                            class_id = None
                                            cv2.imwrite(os.path.join(save_frame_path, f'{subject}-{str(tmp_class_id).zfill(3)}-{view}-{video_path.split(".")[0]}-{number_generated_geis}.jpg'), frame)
                                            cv2.imwrite(os.path.join(save_gei_path, f'{subject}-{str(tmp_class_id).zfill(3)}-{view}-{video_path.split(".")[0]}-{number_generated_geis}.jpg'), gei)
                                        #cv2.imshow("rgb", frame)
                            if cv2.waitKey(1) == ord('q'):
                                break
if __name__ == "__main__":
    main()
