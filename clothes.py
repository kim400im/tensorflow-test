import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

(trainX, trainY), (testX, testY) = tf.keras.datasets.fashion_mnist.load_data()

print(trainX[0])
print(trainX.shape)

print(trainY.shape)

plt.imshow(trainX[1])
plt.gray()
plt.colorbar()
# plt.show()

# 데이터 전처리를 하자.
trainX = trainX / 255.0
testX = testX / 255.0

trainX = trainX.reshape( (trainX.shape[0], 28, 28, 1) )
testX = testX.reshape( (testX.shape[0], 28, 28, 1) )

model = tf.keras.Sequential([
    # 필터 크기를 3 * 3으로 한다. 
    # 출력 이미지랑 입력 이미지를 같게 한다.
    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(28,28,1)),
    # 2 * 2 에서 가장 큰걸 찾는다. 
    tf.keras.layers.MaxPooling2D( (2,2) ),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),  # relu는 음수를 내보내지 않는다. 음수는 0으로 처리한다. 
    tf.keras.layers.Dense(10, activation='softmax'),
    # softmax 값을 다 더하면 1이 나온다. 0에서 1 사이로 압축. 카테고리 문제에 적용
    # sigmoid는 0인지 1인지 판단하는 이진 문제에 사용한다. 
])

model.summary()

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

model.fit(trainX, trainY, validation_data=(testX, testY), epochs=5)

score = model.evaluate(testX, testY)
print(score)