"""
This class is responsible for generating test data for appearance-based 
gait recognition. It is used to generate test data for the given views 
and subjects.
"""
from os import path, listdir, system
from typing import List, Dict
from .create_dir import create_dir

class TestDataGenerator:
    """
    A class that generates test data for appearance-based gait recognition.

    Attributes:
        config (Dict[str, str | List]): A dictionary containing the configuration for the test data generator.
        test_clips_dir (str): The directory to store the test clips.
        verbose (bool): Whether to print verbose output.

    Methods:
        process_clips(walk_dir: str, test_dir: str, subject: str, walk: str) -> None:
            Process the clips in the given directory and copy them to the test directory.

        process_walks(subject_dir: str, test_dir: str, subject: str) -> None:
            Process the walks for a given subject.

        process_subjects(test_dir: str, view: str) -> None:
            Process the subjects for a specific view.

        generate_test_data(views: List[str]) -> None:
            Generates test data for the given views.
    """
    def __init__(
        self,
        config: Dict[str, str | List],
        test_clips_dir: str,
        verbose: bool = False,
        ):
        """
        Initializes the TestDataGenerator object.

        Args:
            config (Dict[str, str | List]): A dictionary containing the configuration for the test data generator.
            test_clips_dir (str): The directory to store the test clips.
            verbose (bool): Whether to print verbose output.
        """
        self.clips_directory = config['clips_dir']
        self.test_clips_dir = path.join(config['dataset_dir'], config['dataset_name'], test_clips_dir)
        self.subjects = config['subjects']
        self.nclips = config['test_clips']
        self.views = config['views']
        self.verbose = verbose

    def process_clips(self, walk_dir: str, test_dir: str, subject: str, walk: str) -> None:
        """
        Process the clips in the given directory and copy them to the test directory.

        Args:
            walk_dir (str): The directory containing the clips.
            test_dir (str): The directory where the clips will be copied.
            subject (str): The subject of the clips.
            walk (str): The walk type of the clips.

        Returns:
            None
        """
        clips = sorted(listdir(walk_dir))
        for i, clip in enumerate(clips):
            if clip[-4:] == '.avi' and self.nclips[walk] >= i and i > 5:
                clip_path = path.join(walk_dir, clip)
                sub_dir = path.join(test_dir, subject, walk, clip)
                system(f"cp {clip_path} {sub_dir}")

    def process_walks(self, subject_dir: str, test_dir: str, subject: str) -> None:
        """
        Process the walks for a given subject.

        Args:
            subject_dir (str): The directory path of the subject's data.
            test_dir (str): The directory path where the test data will be generated.
            subject (str): The name of the subject.

        Returns:
            None
        """
        for walk in ['nm', 'bg', 'cl']:
            create_dir(path.join(test_dir, subject, walk), force=True)
            walk_dir = path.join(subject_dir, walk)
            self.process_clips(walk_dir, test_dir, subject, walk)

    def process_subjects(self, test_dir: str, view: str) -> None:
        """
        Process the subjects for a specific view.

        Args:
            test_dir (str): The directory where the test data will be generated.
            view (str): The specific view to process.

        Returns:
            None
        """
        for subject in self.subjects:
            if self.verbose:
                print(f'Processing subject: {subject} view: {view}')
            subject_dir = path.join(self.clips_directory, view, subject)
            self.process_walks(subject_dir, test_dir, subject)

    def generate_test_data(self) -> None:
        """
        Generates test data for the given views.

        Args:
            views (List[str]): List of views for which test data needs to be generated.

        Returns:
            None
        """
        for view in self.views:
            test_dir = path.join(self.test_clips_dir, view)
            if self.verbose:
                print(f'PROCESSING VIEW: {view}')
            self.process_subjects(test_dir, view)
