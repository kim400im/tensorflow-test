# import tensorflow as tf
# if tf.test.gpu_device_name():
#     print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

# import tensorflow as tf
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# import tensorflow as tf
# print(tf.__version__)

import os
import tensorflow as tf 
from tensorflow.python.client import device_lib
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print(device_lib.list_local_devices() )


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
