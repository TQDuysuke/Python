import tensorflow as tf
from tensorflow import keras
import numpy as np

# Dữ liệu cố định
X_train = np.array([[10], [20], [30], [40], [60], [70], [80], [90], [100]], dtype=np.float32)
y_train = (X_train > 50).astype(int)  # Gán nhãn 1 nếu giá trị > 50, ngược lại 0

# Xây dựng mô hình
model = keras.Sequential([
    keras.layers.Dense(10, activation='relu', input_shape=(1,)),
    keras.layers.Dense(1, activation='sigmoid')
])

# Biên dịch mô hình
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình
model.fit(X_train, y_train, epochs=5000)

# Chuyển đổi mô hình TensorFlow sang TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Lưu mô hình TensorFlow Lite
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
