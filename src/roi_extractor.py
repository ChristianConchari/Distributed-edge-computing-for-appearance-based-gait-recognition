"""
This class is used to extract the region of interest (ROI) from frames for each view and subject.
"""
from typing import List
from os import path, listdir
from cv2 import imread, cvtColor, COLOR_BGR2RGB, imwrite # pylint: disable=no-name-in-module
from .roi_finder import ROIFinder
from .create_dir import create_dir

class ROIExtractor:
    """
    Class for extracting the region of interest (ROI) from frames for each view and subject.
    
    Attributes:
        images_dir (str): The directory path where the images are stored.
        region_of_interest_dir (str): The directory path where ROIs will be saved.
        frs_dir (str): The directory name where the frames are stored.
        views (List[str]): A list of view identifiers.
        subjects (List[str]): A list of subject identifiers.
        verbose (bool): A flag indicating whether to print verbose output.
        model_path (str): The path to the model used for ROI extraction.
        device (str): The device to be used for ROI extraction (e.g., 'cpu', 'cuda').
        walk_types (List[str]): A list of walk types.
        roi_finder (ROIFinder): An instance of the ROIFinder class.
    
    Methods:
        process_image: Processes an image by extracting the region of interest (ROI) 
                       and saving it as a separate image file.
        process_sequence: Processes a sequence of images to extract regions of interest (ROIs).
        process_walk: Processes a walk for a given subject.
        process_subject: Processes the subject by creating directories for ROIs 
                       and processing each walk type.
        extract_roi_from_frames: Extracts the region of interest (ROI) from frames 
                       for each view and subject.
    """
    def __init__(
        self,
        images_dir: str,
        region_of_interest_dir: str,
        frs_dir: str,
        views: List[str],
        subjects: List[str],
        roi_finder: ROIFinder,
        verbose: bool,
        walk_types: List[str] = None
    ):
        """
        Initializes an instance of the ROIExtractor class.
        """
        self.images_dir = images_dir
        self.region_of_interest_dir = region_of_interest_dir
        self.frs_dir = frs_dir
        self.views = views
        self.subjects = subjects
        self.verbose = verbose
        self.walk_types = walk_types or ['nm', 'bg', 'cl']
        self.roi_finder = roi_finder

    def process_image(
        self,
        image_path: str,
        sub_roi_dir: str,
        subject: str,
        seq: str,
        cnt: int,
        seq_frames_dir: str
        ) -> int:
        """
        Processes an image by extracting the region of interest (ROI) 
        and saving it as a separate image file.

        Args:
            image_path (str): The path of the image file.
            sub_roi_dir (str): The directory where the extracted ROIs will be saved.
            subject (str): The subject identifier.
            seq (str): The sequence identifier.
            cnt (int): The current count of processed images.
            seq_frames_dir (str): The directory where the sequence frames are stored.

        Returns:
            int: The updated count of processed images.
        """
        rgb = imread(path.join(seq_frames_dir, image_path))
        try:
            bbox = self.roi_finder.find_roi(rgb)
            roi = cvtColor(rgb[bbox[1]:bbox[3], bbox[0]:bbox[2]], COLOR_BGR2RGB)
            roi_name = path.join(sub_roi_dir, f'{subject}-{seq}-{str(cnt).zfill(2)}.jpg')
            imwrite(roi_name, roi)
            return cnt + 1
        except Exception as e: # pylint: disable=broad-except
            if self.verbose:
                print(e)
            return cnt

    def process_sequence(
        self,
        seq_dir: str,
        rois_dir: str,
        subject: str,
        walk: str,
        seq: str
        ) -> None:
        """
        Processes a sequence of images to extract regions of interest (ROIs).

        Args:
            seq_dir (str): The directory path of the sequence of images.
            rois_dir (str): The directory path to save the extracted ROIs.
            subject (str): The subject identifier.
            walk (str): The walk identifier.
            seq (str): The sequence identifier.

        Returns:
            None
        """
        sub_roi_dir = path.join(rois_dir, subject, walk, seq)
        create_dir(sub_roi_dir, force=True)
        seq_frames_dir = path.join(seq_dir, seq)
        lfiles = sorted(listdir(seq_frames_dir))
        cnt = 0
        for image_path in lfiles:
            cnt = self.process_image(image_path, sub_roi_dir, subject, seq, cnt, seq_frames_dir)

    def process_walk(self, frames_dir: str, rois_dir: str, subject: str, walk: str) -> None:
        """
        Process a walk for a given subject.

        Args:
            frames_dir (str): The directory path where the frames are stored.
            rois_dir (str): The directory path where the ROIs will be saved.
            subject (str): The subject identifier.
            walk (str): The walk identifier.

        Returns:
            None
        """
        seq_dir = path.join(frames_dir, subject, walk)
        seqs = sorted(listdir(seq_dir))
        for seq in seqs:
            self.process_sequence(seq_dir, rois_dir, subject, walk, seq)

    def process_subject(self, frames_dir: str, rois_dir: str, subject: str) -> None:
        """
        Process the subject by creating directories for ROIs and processing each walk type.

        Args:
            frames_dir (str): The directory path where the frames are stored.
            rois_dir (str): The directory path where the ROIs will be saved.
            subject (str): The subject identifier.

        Returns:
            None
        """
        if self.verbose:
            print(f'Processing subject: {subject}')
        create_dir(path.join(rois_dir, subject), force=True)
        for walk in self.walk_types:
            self.process_walk(frames_dir, rois_dir, subject, walk)

    def extract_roi_from_frames(self) -> None:
        """
        Extracts the region of interest (ROI) from frames for each view and subject.

        This method iterates over each view and subject, and calls the `process_subject` method
        to extract the ROI from frames for that subject.

        Returns:
            None
        """
        for view in self.views:
            frames_dir = path.join(self.images_dir, view, self.frs_dir)
            rois_dir = path.join(self.region_of_interest_dir, view)
            if self.verbose:
                print(f'PROCESSING VIEW: {view}')
            subjects = sorted(listdir(frames_dir))
            for subject in subjects:
                self.process_subject(frames_dir, rois_dir, subject)
