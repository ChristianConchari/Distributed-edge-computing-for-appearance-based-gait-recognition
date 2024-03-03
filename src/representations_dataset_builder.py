"""
This module contains the DatasetBuilder class which 
is used to build the dataset for the given representations.
"""
from os import listdir, path
from shutil import copy
from .create_dir import create_dir

class RepresentationsDatasetBuilder:
    """
    A class to build the dataset for the given representations.
    
    This class builds the dataset for the given representations by iterating over the views
    and subjects, and calling the `process_views_and_walks` method to process the views and walks
    for each subject.
    
    Attributes:
        representations_dir (str): The directory path where the representations are stored.
        training_path (str): The path to the training data.
        views (int): The number of views for each sample.
        verbose (bool): Flag indicating whether to display verbose output.
    
    Methods:
        build_datasets: Builds datasets for each view and subject.
        process_views_and_walks: Process the views and walks for a given subject.
        process_gei_images: Process GEI images from a given directory and save them
            in the training directory.
    """
    def __init__(self, representations_dir, training_path, views):
        """
        Initializes a DatasetBuilder object.

        Args:
            representations_dir (str): The directory path where the representations are stored.
            training_path (str): The path to the training data.
            views (int): The number of views for each sample.
        """
        self.representations_dir = representations_dir
        self.training_path = training_path
        self.views = views
        self.verbose = False
        self.create_dirs()
        
    def create_dirs(self):
        """
        Creates directories for training data.

        This method iterates over the views and subjects in the representations directory and creates
        corresponding training directories for each subject.

        Args:
            None

        Returns:
            None
        """
        for view in self.views:
            rep_dir = path.join(self.representations_dir, view)
            subjects = sorted(listdir(rep_dir))
            for subject in subjects:
                sub_training_dir = path.join(self.training_path, subject)
                create_dir(sub_training_dir, force=True)

    def build_datasets(self) -> None:
        """
        Builds datasets for each view and subject.

        This method iterates over the views and subjects, and calls the
        `process_views_and_walks` method to process the views and walks
        for each subject.

        Parameters:
            None

        Returns:
            None
        """
        for view in self.views:
            rep_dir = path.join(self.representations_dir, view)
            subjects = sorted(listdir(rep_dir))
            for subject in subjects:
                self.process_views_and_walks(subject, view, rep_dir)

    def process_views_and_walks(self, subject, view, rep_dir):
        """
        Process the views and walks for a given subject.

        Args:
            subject (str): The subject identifier.
            view (str): The view identifier.
            rep_dir (str): The directory containing the representation images.

        Returns:
            None
        """
        sub_training_dir = path.join(self.training_path, subject)

        for walk in ['nm', 'bg', 'cl']:
            print(f'Processing subject: {subject} view: {view} walk:{walk}')
            sub_gei_dir = path.join(rep_dir, subject, walk)
            self.process_gei_images(sub_gei_dir, sub_training_dir, view, walk)

    def process_gei_images(self, sub_gei_dir, sub_training_dir, view, walk):
        """
        Copy the GEI images from the given directory to the training directory.

        Args:
            sub_gei_dir (str): The directory path containing the GEI images.
            sub_training_dir (str): The directory path to save the processed GEI images.
            view (str): The view of the GEI images.
            walk (str): The walk of the GEI images.
        """
        for dir_name in listdir(sub_gei_dir):
            gei_name = path.join(sub_gei_dir, dir_name)
            gei_image_name_path = path.join(sub_training_dir, f'{view}-{walk}-{dir_name}')
            copy(gei_name, gei_image_name_path)
