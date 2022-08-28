import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt
import tensorflow as tf

# logger to capture errors, warnings, and other information during the build and inference phases
TRT_LOGGER = trt.Logger()

import cv2

def preprocess_image(img_path):
    # read input image
    input_img = cv2.imread(img_path)
    GEI = cv2.resize(input_img, (64, 64))
    GEI = tf.expand_dims(GEI, axis=0)
    
    return GEI

def build_engine(onnx_file_path):
    # initialize TensorRT engine and parse ONNX model
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network()
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # parse ONNX
    with open(onnx_file_path, 'rb') as model:
        print('Beginning ONNX file parsing')
        parser.parse(model.read())
    print('Completed parsing of ONNX file')

    # allow TensorRT to use up to 1GB of GPU memory for tactic selection
    #builder.max_workspace_size = 1 << 30
    # we have only one image in batch
    #builder.max_batch_size = 1
    # use FP16 mode if possible
    if builder.platform_has_fast_fp16:
        builder.fp16_mode = True

    # generate TensorRT engine optimized for the target platform
    print('Building an engine...')
    engine = builder.build_cuda_engine(network)
    context = engine.create_execution_context()
    print("Completed creating Engine")

    return engine, context

def main():
    # initialize TensorRT engine and parse ONNX model
    engine, context = build_engine('saved_model/cnn_gait_recognition.onnx')
    # get sizes of input and output and allocate memory required for input data and for output data
    for binding in engine:
        if engine.binding_is_input(binding):  # we expect only one input
            input_shape = engine.get_binding_shape(binding)
            input_size = trt.volume(input_shape) * engine.max_batch_size * np.dtype(np.float32).itemsize  # in bytes
            device_input = cuda.mem_alloc(input_size)
        else:  # and one output
            output_shape = engine.get_binding_shape(binding)
            # create page-locked memory buffers (i.e. won't be swapped to disk)
            host_output = cuda.pagelocked_empty(trt.volume(output_shape) * engine.max_batch_size, dtype=np.float32)
            device_output = cuda.mem_alloc(host_output.nbytes)

    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()
    
    host_input = np.array(preprocess_image("/home/christian/OAKGait8/test_representations/007/007_090_cl-01.png").numpy(), dtype=np.float32, order='C')
    cuda.memcpy_htod_async(device_input, host_input, stream)

    # run inference
    context.execute_async(bindings=[int(device_input), int(device_output)], stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(host_output, device_output, stream)
    stream.synchronize()

    print(host_output)

if __name__=='__main__':
    main()