"""
This module contains the ROIFinder class, which is used to find 
the region of interest (ROI) in a given image.
"""
from typing import Tuple, Optional
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
        preprocess_image: Preprocesses the input image for further processing.
        get_coordinates: Obtains the coordinates of the ROI for a specific class 
                         in the given result.
        apply_offsets: Applies offsets to the given coordinates and returns 
                         the adjusted coordinates.
        find_roi: Finds the Region of Interest (ROI) in the given image.
    """

    def __init__(self, model_path: str, device: str = "CPU"):
        self.ie = Core()
        self.model = self.ie.read_model(model=model_path)
        self.compiled_ssd = self.ie.compile_model(model=self.model, device_name=device)

    def preprocess_image(self, img: ndarray) -> ndarray:
        """
        Preprocesses the input image for further processing.

        Args:
            img (ndarray): The input image.

        Returns:
            ndarray: The preprocessed image.
        """
        processed_img = resize(cvtColor(img, COLOR_RGB2BGR), (300, 300))
        input_image = dnn.blobFromImage(processed_img, size=(300, 300), ddepth=CV_8U)
        return reshape(input_image, [-1, 3, 300, 300])

    def get_coordinates(
        self,
        result: dict,
        img_shape: Tuple[int, int, int],
        class_id: int,
        ) -> Tuple[int, int, int, int]:
        """
        Obtains the coordinates of the region of interest (ROI) for a 
        specific class in the given result.

        Args:
            result (dict): The result dictionary containing the predicted values.
            img_shape (Tuple[int, int, int]): The shape of the input image.
            class_id (int): The class ID of the object to be detected. Default is 15 (person).

        Returns:
            Tuple[int, int, int, int]: The coordinates of the ROI in the format 
            (x_min, y_min, x_max, y_max).
        """
        values = next(iter(result.values()))
        indices = where(values[0, 0, :, 1] == class_id)
        if indices[0].size > 0:
            first_index = indices[0][0]
            x_min = int(values[0, 0, first_index, 3] * img_shape[1])
            y_min = int(values[0, 0, first_index, 4] * img_shape[0])
            x_max = int(values[0, 0, first_index, 5] * img_shape[1])
            y_max = int(values[0, 0, first_index, 6] * img_shape[0])
            return x_min, y_min, x_max, y_max
        return 0, 0, 0, 0

    def apply_offsets(
        self,
        coords: Tuple[int, int, int, int],
        offsets: Tuple[int, int, int, int],
        img_shape: Tuple[int, int, int]
        ) -> Tuple[int, int, int, int]:
        """
        Applies offsets to the given coordinates and returns the adjusted coordinates.

        Args:
            coords (Tuple[int, int, int, int]): The original coordinates (x, y, x_max, y_max).
            offsets (Tuple[int, int, int, int]): The offsets to be applied (ox, oy, *_).
            img_shape (Tuple[int, int, int]): The shape of the image (height, width, channels).

        Returns:
            Tuple[int, int, int, int]: The adjusted coordinates (x, y, w, h).
        """
        x, y, x_max, y_max = coords
        ox, oy, *_ = offsets
        x = max(x - ox, 0)
        y = max(y - oy, 0)
        w = min(x_max + ox, img_shape[1])
        h = min(y_max + oy, img_shape[0])
        return x, y, w, h

    def find_roi(self, img: ndarray, class_id: Optional[int] = 15) -> Tuple[int, int, int, int]:
        """
        Finds the Region of Interest (ROI) in the given image.

        Args:
            img (ndarray): The input image.
            class_id (Optional[str]): The class ID for filtering the ROI. Default is 15 (person).

        Returns:
            Tuple[int, int, int, int]: The coordinates (x, y, width, height) of the ROI.
        """
        img_orig = copy(img)
        input_image = self.preprocess_image(img)
        result = self.compiled_ssd.infer_new_request({0: input_image})
        x, y, w, h = self.get_coordinates(result, img_orig.shape, class_id)
        offsets = (10, 10, 10, 10)
        return self.apply_offsets((x, y, w, h), offsets, img_orig.shape)
    