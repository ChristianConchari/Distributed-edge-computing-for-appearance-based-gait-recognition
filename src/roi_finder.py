"""
This module contains the ROIFinder class, which is used to find 
the region of interest (ROI) in a given image.
"""
from typing import Tuple
from openvino.runtime import Core
from cv2 import resize, cvtColor, COLOR_RGB2BGR, CV_8U, dnn # pylint: disable=no-name-in-module
from numpy import ndarray, reshape, where, copy

class ROIFinder:
    """
    Class for finding the Region of Interest (ROI) in an image using 
    the the object detection model 
    mobileNet-ssd (https://docs.openvino.ai/latest/omz_models_model_mobilenet_ssd.html) 
    from OpenVINO distribution

    Args:
        model_path (str): The path to the pre-trained model.
        device (str): The device to use for inference. Default is "CPU".

    Attributes:
        ie (Core): The Inference Engine object.
        model: The pre-trained model.
        compiled_ssd: The compiled model for SSD inference.

    Methods:
        find_roi: Finds the Region of Interest (ROI) in the given image.
        _preprocess_image: Preprocesses the input image.
        _infer_ssd_model: Infers the SSD model on the given image.
        _get_bounding_box: Get the bounding box coordinates for a specific class in the result.
        _apply_offsets: Applies offsets to the bounding box coordinates to expand the region of interest.
    """

    def __init__(self, model_path: str, device: str = "CPU"):
        self.ie = Core()
        self.model = self.ie.read_model(model=model_path)
        self.compiled_ssd = self.ie.compile_model(model=self.model, device_name=device)

    def find_roi(self, img: ndarray, class_id: float = 15.) -> Tuple[int, int, int, int]:
        """
        Finds the region of interest (ROI) in the given image.

        Parameters:
        img (ndarray): The input image.
        class_id (float): The class ID to filter the detection results. Default is 15.

        Returns:
        Tuple[int, int, int, int]: The coordinates and dimensions of the ROI (x, y, width, height).
        """
        img_orig = copy(img)
        img = self._preprocess_image(img)
        result = self._infer_ssd_model(img)
        x_min, y_min, x_max, y_max = self._get_bounding_box(result, img_orig, class_id)
        x, y, w, h = self._apply_offsets(x_min, y_min, x_max, y_max, img_orig)
        return x, y, w, h

    def _preprocess_image(self, img: ndarray) -> ndarray:
        """
        Preprocesses the input image

        Args:
            img (ndarray): The input image.

        Returns:
            ndarray: The preprocessed image.
        """
        img = resize(cvtColor(img, COLOR_RGB2BGR), (300, 300))
        input_image = dnn.blobFromImage(img, size=(300, 300), ddepth=CV_8U)
        input_image = reshape(input_image, [-1, 3, 300, 300])
        return input_image

    def _infer_ssd_model(self, img: ndarray) -> dict:
        """
        Infers the SSD model on the given image.

        Args:
            img (ndarray): The input image.

        Returns:
            dict: The inference result.
        """
        result = self.compiled_ssd.infer_new_request({0: img})
        return result

    def _get_bounding_box(
        self,
        result: dict,
        img_orig: ndarray,
        class_id: int
        ) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
        """
        Get the bounding box coordinates for a specific class in the result.

        Parameters:
            result (dict): The result dictionary containing the detection information.
            img_orig (ndarray): The original image.
            class_id (int): The class ID for which to retrieve the bounding box.

        Returns:
            Tuple[ndarray, ndarray, ndarray, ndarray]: The x_min, y_min, x_max, and y_max 
            coordinates of the bounding box.
        """
        class_indices = next(iter(result.values()))[0, 0, :, 1]
        x_min = next(iter(result.values()))[:, :, where(class_indices == class_id), 3] * img_orig.shape[1]
        y_min = next(iter(result.values()))[:, :, where(class_indices == class_id), 4] * img_orig.shape[0]
        x_max = next(iter(result.values()))[:, :, where(class_indices == class_id), 5] * img_orig.shape[1]
        y_max = next(iter(result.values()))[:, :, where(class_indices == class_id), 6] * img_orig.shape[0]
        return x_min, y_min, x_max, y_max

    def _apply_offsets(
        self,
        x_min: ndarray,
        y_min: ndarray,
        x_max: ndarray,
        y_max: ndarray,
        img_orig: ndarray
        ) -> Tuple[int, int, int, int]:
        """
        Applies offsets to the bounding box coordinates to expand the region of interest.

        Args:
            x_min (ndarray): The minimum x-coordinate of the bounding box.
            y_min (ndarray): The minimum y-coordinate of the bounding box.
            x_max (ndarray): The maximum x-coordinate of the bounding box.
            y_max (ndarray): The maximum y-coordinate of the bounding box.
            img_orig (ndarray): The original image.

        Returns:
            Tuple[int, int, int, int]: The updated coordinates of the bounding box.
        """
        x, y, w, h = int(x_min), int(y_min), int(x_max), int(y_max)
        offsets = [10, 10, 10, 10]
        x = x - offsets[0] if x - offsets[0] >= 0 else 0
        y = y - offsets[3] if y - offsets[3] >= 0 else 0
        w = w + offsets[1] + offsets[0] if w + offsets[1] + offsets[0] <= img_orig.shape[1] else img_orig.shape[1]
        h = h + offsets[3] + offsets[1] if h + offsets[3] + offsets[1] <= img_orig.shape[0] else img_orig.shape[0]
        return x, y, w, h
