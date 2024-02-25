"""
This file contains the code for running the online inference for 
the gait recognition system. This code is only for testing purposes
in the host machine. The code is not intended to be run in any edge 
node.
"""
import sys
import os
sys.path.insert(0, '../../')
from src.create_dir import create_dir
from src.roi_finder import ROIFinder
from src.silhouette_segmenter import SilhouetteSegmenter

from datetime import datetime

import csv
import cv2
import numpy as np
import tensorflow as tf

def main():
    # load classification model
    cnn_gait_recognition_model = tf.saved_model.load("models/cnn_gait_recognition_acc_0.9514_loss_0.3384_val_acc_0.9361_loss_acc_0.4261_TFTRT_FP16")
    # initilize the roi finder
    roi_finder = ROIFinder(model_path="models/public/mobilenet-ssd/FP32/mobilenet-ssd.xml", device="CPU")
    # initialize the segmentation model
    segmenter = SilhouetteSegmenter(model_path="../../models/128x128_acc_0.8786_loss_0.1018_val-acc_0.8875_val-loss_0.0817_0.22M_13-09-22-TF_OAKGait16.xml", device="CPU")
    # define test clips directory
    test_clips = "Offline-processing/Datasets/OakGait16/test_clips"
    # get all the views in the test clips directory
    views = sorted(os.listdir(test_clips))
    # iterate over all the views
    for view in views:
        subjects = sorted(os.listdir(os.path.join(test_clips,view)))
        # iterate over all the subjects
        for subject in subjects:
            # get all the walks in the subject directory
            walks = sorted(os.listdir(os.path.join(test_clips,view,subject)))
            # define the path to save the frames
            save_frame_path = os.path.join("Offline-processing/Datasets/OakGait16/test_clips", subject, view)
            # define the path to save the geis
            save_gei_path = os.path.join("Offline-processing/Datasets/OakGait16/test_clips", subject, view, "GEIs")
            # create the directories
            create_dir(save_frame_path)
            create_dir(save_gei_path)
            # iterate over all the walks
            for walk in walks:
                # get all the video files in the walk directory
                video_files = sorted(os.listdir(os.path.join(test_clips,view,subject,walk)))
                correct_classification = 0
                number_generated_geis = 0
                # iterate over all the video files
                for video_path in video_files:
                    # get the current time
                    now = datetime.now()
                    # initialize the color for the bounding box
                    bbox_color = (0, 0, 255)
                    # initialize the silhouettes list
                    silhouettes = []
                    # initialize the class ID
                    class_id = None
                    # initialize a black image to display the silhouette
                    sil_centered = np.zeros((220,220), np.uint8)
                    # initialize a black image to display the GEI
                    gei = np.zeros((220,220), np.uint8)
                    # print the current video being processed
                    print(f"Processing: subject {subject} in view {view} in walk {walk}")
                    # define the video capture object
                    cap = cv2.VideoCapture(os.path.join(test_clips, view, subject, walk, video_path))
                    # get the length of the video
                    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    # if the length is greater than 85, set the cfr to 85, else set it to 70
                    if length > 85: 
                        cfr = 85
                    elif subject == '005':
                        cfr = 70
                    else:
                        cfr = int(length*0.85)
                    # iterate over all the frames in the video
                    while cap.isOpened():
                        # read the frame
                        read_correctly, frame = cap.read()
                        # define a copy of the frame for modification
                        rgb_image = frame.copy()
                        # if the frame is not read correctly, break the loop
                        if not read_correctly:
                            break
                        # get the bouding box of the ROI
                        x, y, w, h = roi_finder.find_roi(rgb_image)
                        # convert the ROI to RGB
                        roi = cv2.cvtColor(rgb_image[y:h, x:w], cv2.COLOR_BGR2RGB)
                        try:
                            # get the height of the ROI
                            roi_height = roi.shape[0]
                            # get the width of the ROI
                            roi_width = roi.shape[1]
                            # get the silhouette of the ROI
                            sil = cv2.resize(segmenter.sil_segmentation(roi), (roi_width,roi_height)) 
                            # if the mean of the silhouette is greater than 2, get the bounding box
                            if sil.mean() > 2:
                                # get the bounding box of the silhouette
                                x, y, w, h = cv2.boundingRect(sil)
                                # get the centered silhouette
                                sil_centered = segmenter.sil_centering(sil, 220)
                        except Exception as e: # pylint: disable=broad-except
                            print(e)
                            sil_centered = np.zeros((220,220), np.uint8)
                        # append normalized silhouettes to list
                        silhouettes.append(sil_centered)
                        # compute gei for 40 silhouettes, approx. 1.33 seconds of walking at approx. 30 FPS
                        if len(silhouettes) % cfr == 0:
                            try:
                                # Compute the GEI
                                gei = np.mean(np.array(silhouettes), axis=0).astype('uint8')
                                # if the mean of the GEI is less than 10, set the class ID to None
                                if np.mean(gei) < 10: 
                                    class_id = None
                                # normalize the GEI
                                gei_pred = gei/255.0
                                # reshape the GEI to the model input shape
                                gei_pred = tf.reshape(gei_pred, [-1, 220, 220, 1])
                                # cast the GEI to float32
                                gei_pred = tf.dtypes.cast(gei_pred, tf.float32)
                                # increment the number of generated GEIs
                                number_generated_geis += 1
                                # run inference
                                prediction = np.asarray(cnn_gait_recognition_model.predict(gei_pred))
                                # get the class id 
                                class_id = int(tf.argmax(prediction, axis = 1))
                                # get the prediction accuracy
                                pred_acc = round(prediction[0][class_id], 2)
                                # check if identified subject correspond to video subject
                                if str(class_id).zfill(3) == subject:
                                    correct_classification += 1
                                    bbox_color = (0, 255, 0)
                                else:
                                    bbox_color = (0, 0, 255)
                                # show the GEI
                                cv2.imshow("GEI", gei)
                                # save relevant information
                                row = [now.strftime("%m-%d-%Y--%H-%M") + "," + str(class_id).zfill(3) + "," + subject + "," + walk + "," + str(pred_acc) ]
                                # save the relevant information to a csv file
                                with open(f'data_log_{view}.csv', 'a', encoding='utf-8') as f:
                                    # create the csv writer
                                    writer = csv.writer(f)
                                    # write a row to the csv file
                                    writer.writerow(row)
                            except Exception as e: # pylint: disable=broad-except
                                print(e)
                        # clean silhouettes array if 40 silhouettes is reached
                        if len(silhouettes) > cfr:
                            silhouettes = []
                        # if subject is not identified show wait message, else show subject ID                                    
                        if class_id is None:
                            text_id = "wait"
                            bbox_color = (255, 0, 0)
                        else:
                            text_id = str(class_id).zfill(3)
                        # draw bounding box and text
                        cv2.rectangle(frame, (x, y), (x+40, y-20), bbox_color, -1)
                        cv2.putText(frame, text_id, (x + 2, y - 5), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                        cv2.rectangle(frame, (x, y), (w, h), bbox_color, 2)
                        # save identification frames
                        if class_id is not None:
                            cv2.imwrite(os.path.join(save_frame_path, f'{subject}-{str(class_id).zfill(3)}-{view}-{video_path.split(".")[0]}-{number_generated_geis}.jpg'), frame)
                            cv2.imwrite(os.path.join(save_gei_path, f'{subject}-{str(class_id).zfill(3)}-{view}-{video_path.split(".")[0]}-{number_generated_geis}.jpg'), gei)
                        cv2.imshow("rgb", frame)
            if cv2.waitKey(1) == ord('q'):
                break

if __name__ == '__main__':
    main()
