import tensorflow as tf

tensor = tf.constant([3,4,5])
tensor2 = tf.constant([4,5,6])

tensor3 = tf.constant([[1,2], [3,4]])

print(tensor + tensor2)
print(tf.add(tensor, tensor2))

tensor4 = tf.zeros(10)
print(tensor4)

tensor5 = tf.zeros([2,2,3])
print(tensor5)

print(tensor3.shape)

w = tf.Variable(1.0)
w.assign(2)
print(w)