"""
This module contains the SilhouetteSegmenter class, which is used to apply
silhouette segmentation to an image using the silhouette-segmentation-adas-0001 model
from OpenVINO distribution.
"""
from typing import Tuple
from openvino.runtime import Core
from cv2 import resize, findContours, drawContours, contourArea, moments # pylint: disable=no-name-in-module
from cv2 import RETR_EXTERNAL, CHAIN_APPROX_SIMPLE # pylint: disable=no-name-in-module
from numpy import reshape, where, ndarray, zeros_like, concatenate, zeros, array
from numpy import min as npmin, max as npmax

class SilhouetteSegmenter:
    """
    The SilhouetteSegmenter class is responsible for segmenting and processing silhouette images.

    Attributes:
        ie (Core): The Inference Engine object.
        model (Model): The loaded model for silhouette segmentation.
        compiled_model (CompiledModel): The compiled model for inference.
        input_layer (str): The name of the input layer of the model.
        output_layer (str): The name of the output layer of the model.

    Methods:
        __init__(self, model_path: str, device: str = "CPU") -> None:
            Initializes the SilhouetteSegmenter object.
        preprocess_image(self, roi_img: ndarray) -> ndarray:
            Preprocesses the input image by resizing, normalizing, and reshaping it.
        sil_segmentation(self, roi_img: ndarray) -> ndarray:
            Applies silhouette segmentation on the input image.
        sil_centering(self, img: ndarray, size: int = 64) -> ndarray:
            Centers and resizes the silhouette image.
        _crop_silhouette(self, biggest: ndarray, contours1: list) -> Tuple[ndarray, list]:
            Crop the silhouette image based on the given contours.
        _resize_silhouette(self, silhouette: ndarray, size: int) -> ndarray:
            Resizes the given silhouette image to the specified size.
        _center_silhouette(self, nor_sil: ndarray, size: int) -> ndarray:
            Centers the silhouette image within a background of specified size.
        _crop_and_resize_silhouette(self, biggest: ndarray, contours1: list, size: int) -> ndarray:
            Crop and resize the silhouette image.
    """
    def __init__(self, model_path: str, device: str = "CPU") -> None:
        """
        Initializes the SilhouetteSegmenter object.

        Args:
            model_path (str): The path to the model file.
            device (str, optional): The device name to use for inference. Defaults to "CPU".
        """
        self.ie = Core()
        self.model = self.ie.read_model(model=model_path)
        self.compiled_model = self.ie.compile_model(model=self.model, device_name=device)
        self.input_layer = next(iter(self.compiled_model.inputs))
        self.output_layer = next(iter(self.compiled_model.outputs))

    def preprocess_image(self, roi_img: ndarray) -> ndarray:
        """
        Preprocesses the input image by resizing it to (128, 128), 
        normalizing pixel values to [0, 1], and reshaping it 
        to match the expected input shape of the model.

        Args:
            roi_img (ndarray): The input image.

        Returns:
            ndarray: The preprocessed image.
        """
        input_image = resize(roi_img, (128, 128))
        input_image = input_image / 255.0
        return reshape(input_image, [1, 128, 128, 3])

    def sil_segmentation(self, roi_img: ndarray) -> ndarray:
        """
        Applies silhouette segmentation on the input image.

        Args:
            roi_img (ndarray): The input image.

        Returns:
            ndarray: The segmented image.
        """
        input_image = self.preprocess_image(roi_img)
        # Perform inference
        result = self.compiled_model.infer_new_request({self.input_layer.any_name: input_image})
        # Transpose the output to match the input shape
        prediction = next(iter(result.values())).transpose(0, 3, 1, 2)
        # Convert the output to a binary image
        return array(where(prediction[0][0] > 0.5, 255, 0)).astype('uint8')

    def sil_centering(self, img: ndarray, size: int = 64) -> ndarray:
        """
        Centers and resizes the silhouette image.

        Args:
            img (ndarray): The input silhouette image.
            size (int, optional): The desired size of the output image. Defaults to 64.

        Returns:
            ndarray: The centered and resized silhouette image.
        """
        # Find the largest contour
        biggest = zeros_like(img)
        contours1, _ = findContours(img.copy(), RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
        cnt = max(contours1, key=contourArea)
        drawContours(biggest, [cnt], -1, 255, -1)
        if len(contours1) > 0:
            # Crop and resize the silhouette
            silhouette = self._crop_and_resize_silhouette(biggest, contours1, size)
            return silhouette
        return zeros((size, size))

    def _crop_silhouette(self, biggest: ndarray, contours1: list) -> Tuple[ndarray, list]:
        """
        Crop the silhouette image based on the given contours.

        Args:
            biggest (ndarray): The biggest contour image.
            contours1 (list): List of contours.

        Returns:
            Tuple[ndarray, list]: The cropped silhouette image and the bounding 
                box coordinates [x1, y1, x2, y2].
        """
        ncoun = concatenate(contours1)[:, 0, :]
        x1, y1 = npmin(ncoun, axis=0)
        x2, y2 = npmax(ncoun, axis=0)
        silhouette = biggest[y1:y2, x1:x2]
        return silhouette, [x1, y1, x2, y2]

    def _resize_silhouette(self, silhouette: ndarray, size: int) -> ndarray:
        """
        Resizes the given silhouette image to the specified size.

        Args:
            silhouette (ndarray): The input silhouette image.
            size (int): The desired size of the output silhouette image.

        Returns:
            ndarray: The resized silhouette image.
        """
        factor = size / max(silhouette.shape)
        height = round(factor * silhouette.shape[0])
        width = round(factor * silhouette.shape[1])
        return resize(silhouette, (width, height))

    def _center_silhouette(self, nor_sil: ndarray, size: int) -> ndarray:
        """
        Centers the silhouette image within a background of specified size.

        Args:
            nor_sil (ndarray): The normalized silhouette image.
            size (int): The size of the background.

        Returns:
            ndarray: The centered silhouette image with the specified background size.
        """
        portion_body = 0.3
        sil_moments = moments(nor_sil[0:int(nor_sil.shape[0] * portion_body),])
        w = round(sil_moments['m10'] / sil_moments['m00'])
        background = zeros((size, size))
        shift = round((size / 2) - w)
        if shift < 0 or shift + nor_sil.shape[1] > size:
            shift = round((size - nor_sil.shape[1]) / 2)
        background[:, shift:nor_sil.shape[1] + shift] = nor_sil
        return background

    def _crop_and_resize_silhouette(self, biggest: ndarray, contours1: list, size: int) -> ndarray:
        """
        Crop and resize the silhouette image.

        Args:
            biggest (ndarray): The biggest contour of the silhouette.
            contours1 (list): List of contours of the silhouette.
            size (int): The desired size of the resized silhouette.

        Returns:
            ndarray: The cropped and resized silhouette image.
        """
        silhouette, _ = self._crop_silhouette(biggest, contours1)
        nor_sil = self._resize_silhouette(silhouette, size)
        return self._center_silhouette(nor_sil, size)
    