import tensorflow as tf
print("GPU available:", tf.test.is_gpu_available())
print("Built with GPU support:", tf.test.is_built_with_cuda())