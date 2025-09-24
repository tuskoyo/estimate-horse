import tensorflow as tf
import os

# デバッグ情報を表示
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
print("CUDA_VISIBLE_DEVICES:", os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set'))
print("CUDA_HOME:", os.environ.get('CUDA_HOME', 'Not Set'))
print("LD_LIBRARY_PATH:", os.environ.get('LD_LIBRARY_PATH', 'Not Set'))

# TensorFlowのCUDA情報
print("TensorFlow version:", tf.__version__)
print("CUDA enabled:", tf.test.is_built_with_cuda())
print("GPU available:", tf.config.list_physical_devices('GPU'))
