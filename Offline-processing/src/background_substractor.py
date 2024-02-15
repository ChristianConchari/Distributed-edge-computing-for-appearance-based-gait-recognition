"""
This class is used to perform background substraction using the Lab color space
"""
from typing import Tuple, List
from colorsys import hsv_to_rgb

import cv2
import numpy as np

class BackgroundSubstractor():
    """
    This class is used to perform background substraction using the Lab color space
    
    Attributes:
        kernel: The kernel used for morphological operations
        color: The color used for visualization
    
    Methods:
        _is_empty: Checks if in the current frame there is someone using the histogram of a blank laboratory
        _find_mask: Takes a raw mask as input and returns the biggest contour mask
        _random_colors: Generate random colors for visualization
        bkgrd_lab_substraction: Performs background substraction using the Lab color space
    """
    def __init__(self):
        self.kernel = np.ones((5,5),np.uint8)
        self.color = self._random_colors(1)[0]
        
    def _is_empty(self, scene: np.ndarray) -> bool:
        ''' 
        Checks if in the current frame there is someone using the histogram of a blank laboratory
        
        Args:
            scene: The current frame to be checked
        
        Returns:
            bool: True if the frame is empty, False otherwise
        '''
        hsv_scene = cv2.cvtColor(scene, cv2.COLOR_BGR2HSV)
        h, s, _ = cv2.split(hsv_scene)
        h_array = h[s > 100].flatten()
        val, _ = np.histogram(h_array, 18)
        val[[0,3]] = 0

        return val.max()<1000
        
    def _find_mask(self, mask: np.ndarray) -> np.ndarray:
        ''' 
        Takes a raw mask as input and returns the biggest contour mask
        
        Args:
            mask: The raw mask to be processed
        
        Returns:
            out: The biggest contour mask
        '''
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(contours, key = cv2.contourArea)
        out = np.zeros_like(mask)
        out = cv2.drawContours(out, [c], 0, 255, -1)
        return out


    def _random_colors(self, n: int, bright: bool=True) -> List[Tuple[int, int, int]]:
        ''' 
        Generate random colors for visualization.

        Args:
            n (int): The number of colors to generate.
            bright (bool, optional): Flag to indicate whether to generate
                bright colors. Defaults to True.

        Returns:
            List[Tuple[int, int, int]]: A list of n colors in RGB format.
        '''
        brightness = 1.0 if bright else 0.7
        hsv = [(i / n, 1, brightness) for i in range(n)]
        colors = list(map(lambda c: hsv_to_rgb(*c), hsv))
        return colors
    
    def bkgrd_lab_substraction(
        self,
        frame: np.ndarray,
        background: np.ndarray,
        th: int=5,
        b_th: int=5
        ) -> Tuple[bool, np.ndarray, np.ndarray, np.ndarray]:
        ''' 
        Performs background substraction using the Lab color space, 
        which helps to reduce light noise
        
        Args:
            frame: The current frame to be processed
            background: The background to be substracted
            th: The threshold to consider the frame as empty
            b_th: The threshold to consider the difference between the background and the frame
        
        Returns:
            Tuple[bool, np.ndarray, np.ndarray, np.ndarray]: A tuple with the following elements:
                - A boolean indicating if the frame is empty
                - The raw mask
                - The biggest contour mask
                - The blured mask
        '''
        bkgrd_h = cv2.cvtColor(background, cv2.COLOR_BGR2Lab)
        frgrd_h = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)

        diff = cv2.subtract(bkgrd_h, frgrd_h)
        diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, diff = cv2.threshold(diff, b_th, 255, cv2.THRESH_BINARY)

        diff = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, self.kernel,1)
        diff = cv2.morphologyEx(diff, cv2.MORPH_DILATE, self.kernel,2)
        if diff.sum() / 1000000 > th:
            closed = self._find_mask(diff)
            blured = cv2.GaussianBlur(closed,(3,3),0)
            closed = cv2.multiply(diff, closed)
            return True, diff, closed, blured
        return False, None, None, None
