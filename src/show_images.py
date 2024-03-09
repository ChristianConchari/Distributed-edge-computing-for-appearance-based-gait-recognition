"""
This module contains the function to show images from the given directory.
"""
from os import path, listdir
from random import sample
from typing import List, Optional, Dict
from cv2 import imread, cvtColor, COLOR_BGR2RGB # pylint: disable=no-name-in-module
from numpy import ndarray
from matplotlib.pyplot import imshow, axis, title, show, figure, subplot

def _load_and_convert_image(image_path: str) -> ndarray:
    """
    Loads an image from the given image_path and converts it to RGB format.

    Parameters:
    - image_path (str): The path to the image file.

    Returns:
    - numpy.ndarray: The loaded image in RGB format.
    """
    try:
        image = imread(image_path)
        if image is None:
            raise IOError(f"Image not found or unable to open: {image_path}")
        rgb_image = cvtColor(image, COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error loading or converting image: {e}")
        raise
    return rgb_image

def _plot_images_from_sequences(rseqs_dir: str, nplot: int) -> int:
    """
    Plots images from sequences in the given directory.
    
    Parameters:
    - rseqs_dir (str): The directory containing the sequences.
    
    Returns:
    - int: The updated count of plotted images.
    """
    try:
        seqs = sorted(listdir(rseqs_dir))
        for seq in seqs:
            images = sorted(listdir(path.join(rseqs_dir, seq)))
            image_path = path.join(rseqs_dir, seq, images[30])  # Example; consider parameterizing
            rgb_image = _load_and_convert_image(image_path)
            subplot(3, 5, nplot)
            imshow(rgb_image)
            title(seq)
            axis('off')
            nplot += 1
    except Exception as e: # pylint: disable=broad-except
        print(f"Failed to plot images from sequences: {e}")
    return nplot

def _process_subjects(frames_dir: str, num_subjects: int) -> None:
    """
    Process a specified number of randomly picked subjects in the given frames directory.
    
    Attributes:
    - frames_dir (str): The directory containing the frames.
    - num_subjects (int): The number of subjects to process.

    Returns:
    - None
    """
    try:
        subjects = sorted(listdir(frames_dir))
        random_subjects = sample(subjects, num_subjects)
        for subject in random_subjects:
            print(f'Showing subject: {subject}')
            figure(figsize=(20, 4.5))
            nplot = 1
            for j, walk in enumerate(['nm', 'bg', 'cl']):
                nplot = j*5 + 1
                rseqs_dir = path.join(frames_dir, subject, walk)
                nplot = _plot_images_from_sequences(rseqs_dir, nplot)
            show()
    except Exception as e: # pylint: disable=broad-except
        print(f"Failed to process subjects: {e}")

def show_images(
    config: Dict[str, str | List],
    images_dir: str,
    num_subjects: int = 3
    ) -> None:
    """
    Process multiple views of gait images.
    
    Parameters:
    - config (dict[str | List]): The configuration dictionary.
    - images_dir (str): The directory containing the images.
    - num_subjects (int): The number of subjects to process.
    
    Returns:
    - None
    """
    try:
        for view in config['views']:
            frames_dir = path.join(images_dir, view)
            print(f'SHOWING VIEW: {view}')
            _process_subjects(frames_dir, num_subjects)
    except Exception as e: # pylint: disable=broad-except
        print(f"Failed to show images: {e}")
