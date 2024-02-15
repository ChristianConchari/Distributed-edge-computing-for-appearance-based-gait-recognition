"""
This module is used to extract the masks from the video clips.
"""
from typing import List, Dict
from os import path, listdir

import cv2
import numpy as np

from .create_dir import create_dir
from .background_substractor import BackgroundSubstractor

class MaskExtractor:
    """
    This class is used to extract the masks from the video clips.
    
    Attributes:
        images_dir (str): The directory to save the extracted frames.
        clips_directory (str): The directory containing the video clips.
        mks_dir (str): The directory to save the generated masks.
        frs_dir (str): The directory to save the processed frames.
        subjects (List[str]): The list of subjects.
        views (List[str]): The list of views.
        nclips (Dict): The number of clips for each walk.
        verbose (bool): If True, prints the exceptions raised during the process.
    
    Methods:
        _is_valid_frame: Checks if a frame is valid based on the sum of pixel values in the upper and lower halves of the frame.
        process_clip: Processes a video clip by applying background subtraction and saving the resulting frames and masks.
        _process_clips: Process the video clips by extracting frames, creating directories for frames and masks, and calling
            the `process_clip` method for each clip.
        _process_subjects: Process the subjects for a given view.
        extract_masks: Subtracts the background
    """
    def __init__(self,
        images_dir: str,
        clips_directory: str,
        mks_dir: str,
        frs_dir: str,
        subjects: List[str],
        views: List[str],
        nclips: Dict,
        verbose: bool=False
        ):
        self.images_dir = images_dir
        self.clips_directory = clips_directory
        self.mks_dir = mks_dir
        self.frs_dir = frs_dir
        self.subjects = subjects
        self.views = views
        self.nclips = nclips
        self.verbose = verbose
        self.background_substractor = BackgroundSubstractor()
    
    def _is_valid_frame(self, closed):
        """
        Checks if a frame is valid based on the sum of pixel values in the upper and lower halves of the frame.

        Parameters:
        closed (numpy.ndarray): The binary image representing the frame.

        Returns:
        bool: True if the frame is valid, False otherwise.
        """
        return np.sum(closed[540 // 2 + 40:, :]) > 3500000 and np.sum(closed[:540 // 2 + 40, :]) > 2000000

    def _process_clip(self, subject: str, clip_path: str, sub_dir: str, mask_dir: str, background: np.ndarray):
        """
        Processes a video clip by applying background subtraction and saving the resulting frames and masks.

        Args:
            clip_path (str): The path to the video clip.
            sub_dir (str): The directory to save the processed frames.
            mask_dir (str): The directory to save the generated masks.
            background (numpy.ndarray): The background image used for subtraction.

        Returns:
            None
        """
        cap = cv2.VideoCapture(clip_path)
        cnt = 0
        while True:
            ret, frame = cap.read()
            if ret:
                try:
                    if subject in self.subjects:
                        ok, _, closed, _ = self.background_substractor.bkgrd_lab_substraction(frame, background, b_th=10)
                    else:
                        ok, _, closed, _ = self.background_substractor.bkgrd_lab_substraction(frame, background)
                    if ok:
                        frame_path = f'{sub_dir}/{str(cnt).zfill(4)}.jpg'
                        mask_path = f'{mask_dir}/{str(cnt).zfill(4)}.jpg'
                        frame = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
                        closed = cv2.resize(closed, (960, 540), interpolation=cv2.INTER_AREA)

                        if self._is_valid_frame(closed):
                            cnt += 1
                            cv2.imwrite(frame_path, frame)
                            cv2.imwrite(mask_path, closed)
                except Exception as e:  # pylint: disable=broad-except
                    if self.verbose:
                        print(e)
            else:
                break
            if cv2.waitKey(1) == ord('q'):
                break
        cap.release()
        
    def _process_clips(
        self,
        frames_dir: str,
        masks_dir: str,
        walk: str,
        clips: str,
        walk_dir: str,
        subject: str, 
        background: np.ndarray
        ) -> None:
        """
        Process the video clips by extracting frames, creating directories for frames and masks,
        and calling the `process_clip` method for each clip.

        Args:
            frames_dir (str): The directory to save the extracted frames.
            masks_dir (str): The directory to save the generated masks.
            walk (str): The walk identifier.
            clips (str): The list of video clips.
            walk_dir (str): The directory containing the walk data.
            subject (str): The subject identifier.
            background (np.ndarray): The background image for background subtraction.

        Returns:
            None
        """
        for i, clip in enumerate(clips):
            if clip[-4:] == '.avi' and self.nclips[walk] >= i:
                clip_path = path.join(walk_dir, clip)
                sub_dir = path.join(frames_dir, subject, walk, clip.split('.')[0])
                mask_dir = path.join(masks_dir, subject, walk, clip.split('.')[0])
                create_dir(sub_dir, force=True)
                create_dir(mask_dir, force=True)
                self._process_clip(subject, clip_path, sub_dir, mask_dir, background)
    
    def _process_subjects(self, frames_dir: str, masks_dir: str, view: str):
        """
        Process the subjects for a given view.

        Args:
            frames_dir (str): Directory to save the processed frames.
            masks_dir (str): Directory to save the generated masks.
            view (str): View of the subjects.

        Returns:
            None
        """
        for subject in self.subjects:
            print(f'Processing subject: {subject} view: {view}...')
            create_dir(path.join(frames_dir, subject), force=True)
            create_dir(path.join(masks_dir, subject), force=True)

            subject_dir = path.join(self.clips_directory, view, subject)
            for walk in ['nm', 'bg', 'cl']:
                walk_dir = path.join(subject_dir, walk)
                background = cv2.imread(path.join(walk_dir, 'background.png'))
                clips = sorted(listdir(walk_dir))
                save_back_path = path.join(frames_dir, subject, f'{walk}-background.png')
                cv2.imwrite(save_back_path, background)
                self._process_clips(frames_dir, masks_dir, walk, clips, walk_dir, subject, background)
    
    def extract_masks(self):
        """
        Subtracts the background from the frames in each view.

        This method iterates over each view and processes the frames by subtracting the background.
        It calls the _process_subjects method to perform the background subtraction for each subject in the view.
        Finally, it closes all the windows created by OpenCV.

        Args:
            None

        Returns:
            None
        """
        for view in self.views:
            frames_dir = path.join(self.images_dir, view, self.frs_dir)
            masks_dir = path.join(self.images_dir, view, self.mks_dir)
            print(f'PROCESSING VIEW: {view}')
            self._process_subjects(frames_dir, masks_dir, view)
            print('', end="\r")
        cv2.destroyAllWindows()
