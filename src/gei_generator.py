"""
This module contains the GEIGenerator class,
which is used to generate Gait Energy Images (GEIs) from silhouettes.
"""
from typing import Optional
from os import path, listdir
from cv2 import imread, imwrite # pylint: disable=no-name-in-module
from numpy import mean, array
from .create_dir import create_dir

class GEIGenerator:
    """
    This class is used to generate Gait Energy Images (GEIs) from silhouettes.
    
    Attributes:
        silhouettes_dir (str): The directory containing the silhouettes.
        representations_dir (str): The directory to save the generated representations.
        views (List[str]): The list of views to process.
        walks (List[str]): The list of walks to process.
        verbose (bool): A flag to indicate whether to print verbose output.
        
    Methods:
        extract_gei_from_silhouettes(): 
            Extracts the Gait Energy Image (GEI) from the silhouettes.
        process_view(view: str): 
            Process a specific view of gait data.
        process_subject(subject: str, view: str, silhouettes_directory: str, representations_directory: str): 
            Process a subject's gait data for a specific view.
        process_walk(subject: str, walk: str, silhouettes_directory: str, representations_directory: str): 
            Process a walk for a given subject.
        process_sequence(sequence: str, seq_index: int, sequence_directory: str, subject_gei_directory: str): 
            Process a sequence of frames to generate a gait energy image (GEI).
    """
    def __init__(
        self,
        config,
        silhouettes_dir: str,
        representations_dir: str,
        n_frames: Optional[int] = None,
        verbose: bool = False
        ):
        self.silhouettes_dir = path.join(config['dataset_dir'], config['dataset_name'],  silhouettes_dir)
        self.representations_dir = path.join(config['dataset_dir'], config['dataset_name'], representations_dir)
        self.views = config['views']
        self.walks = config['walks']
        self.n_frames = n_frames
        self.verbose = verbose

    def extract_gei_from_silhouettes(self):
        """
        Extracts the Gait Energy Image (GEI) from the silhouettes.

        This method iterates over the views and processes each view using the `process_view` method.

        Parameters:
            None

        Returns:
            None
        """
        for view in self.views:
            self.process_view(view)

    def process_view(self, view: str):
        """
        Process a specific view of gait data.

        Args:
            view (str): The name of the view.

        Returns:
            None
        """
        silhouettes_directory = path.join(self.silhouettes_dir, view)
        representations_directory = path.join(self.representations_dir, view)
        if self.verbose:
            print(f'GENERATING GAIT REPRESENTATIONS FROM VIEW: {view}')

        subjects = sorted(listdir(silhouettes_directory))
        for subject in subjects:
            self.process_subject(subject, view, silhouettes_directory, representations_directory)

    def process_subject(
        self,
        subject: str,
        view: str,
        silhouettes_directory: str,
        representations_directory: str
        ):
        """
        Process a subject's gait data for a specific view.

        Args:
            subject (str): The subject's identifier.
            view (str): The view identifier.
            silhouettes_directory (str): The directory path where the silhouettes are stored.
            representations_directory (str): The directory path where the representations will be saved.
        """
        if self.verbose:
            print(f'Processing subject: {subject} view: {view}')
        for walk in self.walks:
            self.process_walk(subject, walk, silhouettes_directory, representations_directory)

    def process_walk(
        self,
        subject: str,
        walk: str,
        silhouettes_directory: str,
        representations_directory: str
        ):
        """
        Process a walk for a given subject.

        Args:
            subject (str): The subject identifier.
            walk (str): The walk identifier.
            silhouettes_directory (str): The directory containing the silhouettes.
            representations_directory (str): The directory to store the representations.

        Returns:
            None
        """
        subject_gei_directory = path.join(representations_directory, subject, walk)
        create_dir(subject_gei_directory, force=True)
        sequence_directory = path.join(silhouettes_directory, subject, walk)
        sequences = sorted(listdir(sequence_directory))

        for seq_index, sequence in enumerate(sequences):
            self.process_sequence(sequence, seq_index, sequence_directory, subject_gei_directory)

    def process_sequence(
        self,
        sequence: str,
        seq_index: int,
        sequence_directory: str,
        subject_gei_directory: str
        ):
        """
        Process a sequence of frames to generate a gait energy image (GEI).

        Args:
            sequence (str): The name of the sequence.
            seq_index (int): The index of the sequence.
            sequence_directory (str): The directory containing the frames of the sequence.
            subject_gei_directory (str): The directory to save the generated GEI.

        Returns:
            None
        """
        sequence_frames_directory = path.join(sequence_directory, sequence)
        frame_paths = sorted(listdir(sequence_frames_directory))
        total_frames = len(frame_paths)
        if self.n_frames is None:
            num_batches = 1
            start_index = 0
            end_index = total_frames - 5
        else:
            num_batches = total_frames // self.n_frames  # Compute the max number of GEIs
        for batch_index in range(num_batches):
            start_index = batch_index * self.n_frames
            end_index = min(start_index + self.n_frames, total_frames)
            batch_frame_paths = frame_paths[start_index:end_index]
            
            if len(batch_frame_paths) < self.n_frames:
                continue
    
            silhouettes = [
                imread(path.join(sequence_frames_directory, frame_path))
                for frame_path in batch_frame_paths
            ]
            gei_image = mean(array(silhouettes), axis=0).astype("uint8")
            gei_filename = path.join(subject_gei_directory, f'{str(seq_index+1).zfill(2)}_{batch_index+1}.png')
            imwrite(gei_filename, gei_image)
