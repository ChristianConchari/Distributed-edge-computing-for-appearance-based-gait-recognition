import sys
import os
import cv2
import depthai as dai #pylint: disable=import-error
import time
sys.path.insert(0, '../../')
from src.create_dir import create_dir
from online_pipeline import OnlinePipeline
from datetime import datetime
import csv
import argparse
import numpy as np

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

CNN_GAIT_RECOGNITION_MODEL_PATH = "../../models/tftrt_cnn_gait_recognition_acc_0.9499_loss_0.3443_val_acc_0.8934_loss_acc_0.5382"
MOBILENET_SSD_MODEL_PATH = "../../models/mobilenet-ssd_openvino_2021.4_6shave.blob"
LABEL_MAP = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
SEGMENTATION_MODEL_PATH = "../../models/128x128_acc_0.8433_loss_0.0590_val-acc_0.8462_val-loss_0.0771_0.22M_13-09-22-TF_OAKGait16/model.xml"

TEST_FRAMES_RESULT_PATH = "../../test_frames"
TEST_GEIS_RESULT_PATH = "../../test_GEIs"
VIEW = '075'

def main(subject):
    # initialize the OnlinePipeline class
    op = OnlinePipeline(
        cnn_gait_recognition_model_path=CNN_GAIT_RECOGNITION_MODEL_PATH,
        mobilenet_ssd_model_path=MOBILENET_SSD_MODEL_PATH,
        segmentation_model_path=SEGMENTATION_MODEL_PATH
    )
    silhouettes = []
    class_id = None
    tmp_class_id = None
    out = np.zeros((128,128), np.uint8)
    sil_centered = np.zeros((220,220), np.uint8)
    gei = np.zeros((220,220), np.uint8)
    # create directory if current day directory is not created
    data_recordings = "../../data_recordings"
    csv_data = "../../csv_data"
    gei_path = os.path.join(data_recordings, datetime.now().strftime("%m_%d_%Y") + f"{subject}_{VIEW}_GEI")
    frame_path = os.path.join(data_recordings, datetime.now().strftime("%m_%d_%Y") + f"{subject}_{VIEW}-classID")
    create_dir(data_recordings)
    create_dir(gei_path)
    create_dir(frame_path)
    create_dir(csv_data)
    # create csv file to store data
    with open(f'{csv_data}/data_log_{VIEW}.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["time_stamp", "pred", "label", "gei_number", "ssd_mobile_net_time", "segmentation_time", "cnn_time"])

    # connect to device and start pipeline
    with dai.Device(op.pipeline, op.device_info) as device:
        # output queues will be used to get the rgb frames and nn data from the outputs defined above
        q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        q_det = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

        detections = []
        fps_start_time = time.monotonic()
        counter = 0
        color = (0, 255, 0)
        jpeg_quality = 95
        in_counter = 0
        number_generated_geis = 0
        mobilenet_inference_times = []
        segmentation_inference_times = []

        while True:
            start_time = time.monotonic()
            # get RGB data
            in_rgb = q_rgb.get()
            # get detection data
            in_det = q_det.get()
            end_time = time.monotonic()
            mobilenet_inference_times.append(end_time - start_time)
            counter += 1
            if in_rgb is not None:
                frame = in_rgb.getCvFrame()
            if in_det is not None:
                detections = in_det.detections
            # If the frame is available, draw bounding boxes on it and show the frame
            for detection in detections:
                # If person is detected
                if LABEL_MAP[detection.label] == 'person':
                    # Return false detections counter to zero
                    in_counter = 0
                    # get the roi from the detection
                    roi, bbox = op.get_roi(frame, detection)
                    # get the binarized silhouette
                    start_time = time.monotonic()
                    sil_centered, out = op.get_binarized_silhouette(roi)
                    end_time = time.monotonic()
                    segmentation_inference_times.append(end_time - start_time)
                    # append normalized silhouettes to list
                    silhouettes.append(sil_centered)
                    # compute GEI for 40 silhouettes, 3 seconds of walking and approx. 3 gait cycles
                    if len(silhouettes) % 50 == 0 and len(silhouettes) != 0:
                        data_silhouettes = silhouettes.copy()
                        silhouettes = []
                        number_generated_geis += 1
                        start_time = time.monotonic()
                        class_id, gei = op.get_classification_id(data_silhouettes)
                        end_time = time.monotonic()
                        cnn_inference_time = round(end_time - start_time, 4)
                        row = [
                            datetime.now().strftime("%m-%d-%Y--%H-%M"),
                            str(class_id).zfill(3),
                            subject,
                            str(number_generated_geis),
                            round(sum(mobilenet_inference_times) / len(mobilenet_inference_times), 4),
                            round(sum(segmentation_inference_times) / len(segmentation_inference_times), 4),
                            cnn_inference_time
                        ]
                        print(row)
                        with open(f'../../csv_data/data_log_{VIEW}.csv', 'a', encoding='utf-8', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow(row)
                        # check if identified subject correspond to video subject
                        color = (0, 255, 0) if str(class_id).zfill(3) == subject else (0, 0, 255)

                    # if subject is not identified show wait message, else show subject ID
                    if class_id is None:
                        text_id = "wait"
                        if tmp_class_id is not None:
                            text_id = str(tmp_class_id).zfill(3)
                    else:
                        text_id = str(class_id).zfill(3)
                        
                    
                    cv2.putText(frame, f"NN fps: {counter / (time.monotonic() - fps_start_time):.2f}",
                                (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 1, color)

                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0]+80, bbox[1]-40), color, -1)
                    cv2.putText(frame, text_id, (bbox[0] + 5, bbox[1] - 10), cv2.FONT_HERSHEY_TRIPLEX, 1, (0,0,0), 1, cv2.LINE_AA)
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

                    if class_id is not None:
                        tmp_class_id = class_id
                        class_id = None
                        cv2.imwrite(os.path.join(frame_path, f'{subject}-{str(tmp_class_id).zfill(3)}-{VIEW}-{datetime.now().strftime("%m-%d-%Y--%H-%M-%S")}-{number_generated_geis}.jpg'), frame)
                        cv2.imwrite(os.path.join(gei_path, f'{subject}-{str(tmp_class_id).zfill(3)}-{VIEW}-{datetime.now().strftime("%m-%d-%Y--%H-%M-%S")}-{number_generated_geis}.jpg'), gei)
                else:
                    in_counter += 1
                    if in_counter > 80:
                        silhouettes = []
                        class_id = None
                        gei = np.zeros((160,160), np.uint8)
                        sil_centered = np.zeros((220,220), np.uint8)
                        out = np.zeros((128,128), np.uint8)
                        in_counter = 0

            gei_resized = cv2.resize(gei, (300, 300))
            sil_resized = cv2.resize(out, (300, 300))

            try:
                sil_resized = cv2.cvtColor(sil_resized, cv2.COLOR_GRAY2RGB)
                gei_resized = cv2.cvtColor(gei_resized, cv2.COLOR_GRAY2RGB)
            except Exception as e: # pylint: disable=broad-except
                pass
            # concat important images to send to edge-server
            cimg = op.vconcat_resize([frame, sil_resized, gei_resized])
            cimg = cv2.resize(cimg, (150, 450))
            # define JPG encode parameters
            _, jpg_buffer = cv2.imencode(".jpg", cimg, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
            # send image to edge-server
            op.sender.send_jpg(op.jetson_name, jpg_buffer)
            if cv2.waitKey(1) == ord('q'):
                break
        # desconect imagezmq sender
        op.sender.zmq_socket.disconnect('tcp://'+ op.ip_address+ ':' + op.port)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', type=str, required=True, help='Subject number from dataset OAKGait16')
    main(str(parser.parse_args().subject).zfill(3))
