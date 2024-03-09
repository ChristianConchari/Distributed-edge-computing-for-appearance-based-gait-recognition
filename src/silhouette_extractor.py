"""
This module contains the SilhouetteExtractor class, 
which is used to extract silhouettes from regions of interest (ROIs).
"""
from os import listdir, path
from typing import List, Optional, Dict
from cv2 import imread, imwrite, resize, cvtColor, COLOR_BGR2RGB # pylint: disable=no-name-in-module
from numpy import ndarray

from .create_dir import create_dir
from .silhouette_segmenter import SilhouetteSegmenter

class SilhouetteExtractor:
    """
    A class for extracting silhouettes from regions of interest (ROIs).

    Attributes:
        region_of_interest_dir (str): The directory path of the region of interest (ROI) data.
        silhouettes_dir (str): The directory path to save the extracted silhouettes.
        views (List[str]): The list of views to process.
        segmenter (SilhouetteSegmenter): The silhouette segmenter object.
        walks (List[str], optional): The list of walks to process. Defaults to None.

    Methods:
        extract_silhouettes_from_rois(): Extracts silhouettes from regions of interest (ROIs).
        process_view(view: str): Process a specific view for silhouette extraction.
        process_subject(subject: str, view: str, rois_dir: str, sils_dir: str): 
            Process a subject's gait data for a specific view.
        process_walk(subject: str, walk: str, rois_dir: str, sils_dir: str): 
            Process a walk for a given subject.
        process_sequence(seq: str, subject: str, walk: str, seq_dir: str, sils_dir: str): 
            Process a sequence of images to extract silhouettes.
        process_image(seq_path: str, seq_rois_dir: str, sub_sil_dir: str): 
            Processes an image by segmenting its silhouette and writing it to a directory.
        segment_silhouette(roi: ndarray) -> Optional[ndarray]: 
            Segments the silhouette from the given region of interest (ROI).
        write_silhouette(sil: ndarray, seq_path: str, sub_sil_dir: str): 
            Writes the silhouette image to the specified directory.
    """
    DEFAULT_WALKS = ['nm', 'bg', 'cl']
    SILHOUETTE_THRESHOLD = 2
    IMAGE_WIDTH_FOR_CENTERING = 220

    def __init__(
        self,
        config: Dict[str, str | List],
        region_of_interest_dir: str,
        silhouettes_dir: str,
        segmenter: SilhouetteSegmenter,
        verbose: bool = False
        ):
        """
        Initialize the SilhouetteExtractor class.

        Args:
            region_of_interest_dir (str): 
                The directory path where the region of interest files are located.
            silhouettes_dir (str): The directory path where the silhouette files will be saved.
            views (List[str]): A list of views to process.
            segmenter (SilhouetteSegmenter): An instance of the SilhouetteSegmenter class.
            walks (List[str], optional): A list of walks to process. Defaults to None.
        """
        self.region_of_interest_dir = path.join(config['dataset_dir'], config['dataset_name'], region_of_interest_dir)
        self.silhouettes_dir = path.join(config['dataset_dir'], config['dataset_name'], silhouettes_dir)
        self.views = config['views']
        self.segmenter = segmenter
        self.walks = config['walks']
        self.verbose = verbose

    def extract_silhouettes_from_rois(self) -> None:
        """
        Extracts silhouettes from regions of interest (ROIs).

        This method iterates over the views and processes each view using the `process_view` method.

        Returns:
            None
        """
        for view in self.views:
            self.process_view(view)

    def process_view(self, view: str) -> None:
        """
        Process a specific view for silhouette extraction.

        Args:
            view (str): The name of the view to process.

        Returns:
            None
        """
        rois_dir = path.join(self.region_of_interest_dir, view)
        sils_dir = path.join(self.silhouettes_dir, view)
        if self.verbose:
            print(f'PROCESSING VIEW: {view}')
        subjects = sorted(listdir(rois_dir))

        for subject in subjects:
            self.process_subject(subject, view, rois_dir, sils_dir)

    def process_subject(self, subject: str, view: str, rois_dir: str, sils_dir: str) -> None:
        """
        Process a subject's gait data for a specific view.

        Args:
            subject (str): The subject identifier.
            view (str): The view identifier.
            rois_dir (str): The directory containing the region of interest (ROI) data.
            sils_dir (str): The directory to save the extracted silhouettes.

        Returns:
            None
        """
        if self.verbose:
            print(f'Processing subject: {subject} view: {view}')
        create_dir(path.join(sils_dir, subject), force=True)
        for walk in self.DEFAULT_WALKS:
            self.process_walk(subject, walk, rois_dir, sils_dir)

    def process_walk(self, subject: str, walk: str, rois_dir: str, sils_dir: str) -> None:
        """
        Process a walk for a given subject.

        Args:
            subject (str): The subject identifier.
            walk (str): The walk identifier.
            rois_dir (str): The directory containing the region of interest (ROI) images.
            sils_dir (str): The directory to save the extracted silhouettes.

        Returns:
            None
        """
        seq_dir = path.join(rois_dir, subject, walk)
        seqs = sorted(listdir(seq_dir))
        for seq in seqs:
            self.process_sequence(seq, subject, walk, seq_dir, sils_dir)

    def process_sequence(
        self,
        seq: str,
        subject: str,
        walk: str,
        seq_dir: str,
        sils_dir: str
        ) -> None:
        """
        Process a sequence of images to extract silhouettes.

        Args:
            seq (str): The sequence identifier.
            subject (str): The subject identifier.
            walk (str): The walk identifier.
            seq_dir (str): The directory path of the sequence.
            sils_dir (str): The directory path to save the extracted silhouettes.
        """
        sub_sil_dir = path.join(sils_dir, subject, walk, seq)
        create_dir(sub_sil_dir, force=True)
        seq_rois_dir = path.join(seq_dir, seq)
        lfiles = sorted(listdir(seq_rois_dir))

        for seq_path in lfiles:
            self.process_image(seq_path, seq_rois_dir, sub_sil_dir)

    def process_image(self, seq_path: str, seq_rois_dir: str, sub_sil_dir: str) -> None:
        """
        Processes an image by segmenting its silhouette and writing it to a directory.

        Args:
            seq_path (str): The path of the image file.
            seq_rois_dir (str): The directory containing the image file.
            sub_sil_dir (str): The directory to write the segmented silhouette to.

        Returns:
            None
        """
        roi_name = path.join(seq_rois_dir, seq_path)
        roi = imread(roi_name)
        roi = cvtColor(roi, COLOR_BGR2RGB)
        try:
            sil = self.segment_silhouette(roi)
            if sil is not None:
                self.write_silhouette(sil, seq_path, sub_sil_dir)
        except Exception as e: # pylint: disable=broad-except
            if self.verbose:
                print(f"Error processing image {seq_path}: {e}")

    def segment_silhouette(self, roi: ndarray) -> Optional[ndarray]:
        """
        Segments the silhouette from the given region of interest (ROI).

        Args:
            roi (ndarray): The region of interest (ROI) image.

        Returns:
            Optional[ndarray]: The segmented silhouette image, or None if
            the silhouette is not significant.
        """
        roi_height, roi_width = roi.shape[:2]
        sil = resize(self.segmenter.sil_segmentation(roi), (roi_width, roi_height))
        if sil.mean() > self.SILHOUETTE_THRESHOLD:
            sil_centered = self.segmenter.sil_centering(sil, self.IMAGE_WIDTH_FOR_CENTERING)
            return sil_centered
        return None

    def write_silhouette(self, sil: ndarray, seq_path: str, sub_sil_dir: str) -> None:
        """
        Writes the silhouette image to the specified directory.

        Args:
            sil (ndarray): The silhouette image to be written.
            seq_path (str): The path of the sequence file.
            sub_sil_dir (str): The subdirectory where the silhouette image will be saved.

        Returns:
            None
        """
        sil_name = path.join(sub_sil_dir, f'{seq_path.split(".")[0]}.jpg')
        imwrite(sil_name, sil)
