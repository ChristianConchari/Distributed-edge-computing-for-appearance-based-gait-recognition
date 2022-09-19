import cv2
from openvino.runtime import Core
import numpy as np
from tensorflow import reshape

class Infer_myriad:
    def __init__(self):
        ie = Core()
        device = "MYRIAD.1.2.3-ma2480"
        model = ie.read_model(model="/home/jetson6/models/128x128_acc_0.8786_loss_0.1018_val-acc_0.8875_val-loss_0.0817_0.22M_13-09-22-TF_OAKGait16.xml")
        self.compiled_model = ie.compile_model(model=model, device_name=device)

    def infer(self, roi_img):
        input_image = roi_img/255.0
        input_image = reshape(input_image, [1,128,128,3])
        result = self.compiled_model.infer_new_request({0: input_image})
        prediction = next(iter(result.values())).transpose(0,3,1,2)
        return np.where(prediction[0][0]>0.3, 255, 0).astype('uint8')