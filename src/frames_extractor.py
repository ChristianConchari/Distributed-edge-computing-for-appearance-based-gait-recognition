"""
This class contains the FrameExtractor class, which is responsible for extracting
frames from video clips and saving them as images.
"""
from os import path, listdir
from typing import Dict, List
from cv2 import VideoCapture, imwrite, waitKey, destroyAllWindows # pylint: disable=no-name-in-module
from .create_dir import create_dir

class FrameExtractor:
    """
    This class is responsible for extracting frames from video clips and saving them as images.
    
    attributes:
        config (Dict[str, any]): A dictionary containing the pipeline configuration.
        images_dir (str): The directory where the extracted frames will be saved.
        verbose (bool): Whether to print verbose output.
    
    methods:
        process_clip(clip_path: str, sub_dir: str) -> None:
            Processes a video clip by extracting frames and saving them as images.
        process_walk(walk_dir: str, frames_dir: str, subject: str, walk: str) -> None:
            Process a walk by extracting frames from video clips.
        process_subject(subject_dir: str, frames_dir: str, subject: str) -> None:
            Process the subject's walks in the given subject directory.
        process_view(view: str) -> None:
            Process a specific view for all subjects.
        extract_frames_from_sequences() -> None:
            Extracts frames from sequences for each view.
    """
    def __init__(
        self,
        config: Dict[str, str | List],
        frames_dir: str,
        verbose: bool = False
        ):
        """
        Initializes a FrameExtractor object.

        Args:
            config (Dict[str, any]): A dictionary containing the directories for images, 
                clips, and frames.
            frames_dir (str): The directory where the extracted frames will be saved.
            verbose (bool): Whether to print verbose output.
        """
        self.images_dir = path.join(config['dataset_dir'], config['dataset_name'], frames_dir)
        self.clips_dir= config['clips_dir']
        self.views = config['views']
        self.subjects = config['subjects']
        self.nclips = config['training_clips']
        self.verbose = verbose

    def process_clip(self, clip_path: str, sub_dir: str) -> None:
        """
        Processes a video clip by extracting frames and saving them as images.

        Args:
            clip_path (str): The path to the video clip.
            sub_dir (str): The subdirectory where the frames will be saved.

        Returns:
            None
        """
        cap = VideoCapture(clip_path)
        cnt = 0
        while True:
            ret, frame = cap.read()
            if ret:
                cnt += 1
                frame_path = f'{sub_dir}/{str(cnt).zfill(4)}.jpg'
                imwrite(frame_path, frame)
            else:
                break
            if waitKey(1) == ord('q'):
                break
        cap.release()

    def process_walk(self, walk_dir: str, frames_dir: str, subject: str, walk: str) -> None:
        """
        Process a walk by extracting frames from video clips.

        Args:
            walk_dir (str): The directory containing the video clips of the walk.
            frames_dir (str): The directory where the extracted frames will be saved.
            subject (str): The subject identifier.
            walk (str): The walk identifier.

        Returns:
            None
        """
        clips = sorted(listdir(walk_dir))
        for i, clip in enumerate(clips):
            if clip[-4:] == '.avi' and self.nclips[walk] >= i:
                clip_path = path.join(walk_dir, clip)
                sub_dir = path.join(frames_dir, subject, walk, clip.split('.')[0])
                create_dir(sub_dir, force=True)
                self.process_clip(clip_path, sub_dir)

    def process_subject(self, subject_dir: str, frames_dir: str, subject: str) -> None:
        """
        Process the subject's walks in the given subject directory.

        Args:
            subject_dir (str): The directory path containing the subject's walks.
            frames_dir (str): The directory path to save the extracted frames.
            subject (str): The name of the subject.

        Returns:
            None
        """
        walks = listdir(subject_dir)
        for walk in walks:
            walk_dir = path.join(subject_dir, walk)
            self.process_walk(walk_dir, frames_dir, subject, walk)

    def process_view(self, view: str) -> None:
        """
        Process a specific view for all subjects.

        Args:
            view (str): The view to process.

        Returns:
            None
        """
        frames_dir = path.join(self.images_dir, view)
        for subject in self.subjects:
            if self.verbose:
                print(f'Processing subject: {subject} view: {view}')
            subject_dir = path.join(self.clips_dir, view, subject)
            self.process_subject(subject_dir, frames_dir, subject)

    def extract_frames(self) -> None:
        """
        Extracts frames from sequences for each view.

        This method iterates over each view and calls the `process_view` method to extract frames.
        After processing all views, it closes all windows.

        Returns:
            None
        """
        for view in self.views:
            if self.verbose:
                print(f'PROCESSING VIEW: {view}')
            self.process_view(view)
        destroyAllWindows()
