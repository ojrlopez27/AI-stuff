import tensorflow as tf
import numpy as np

"""
Basic NN that predicts a temperature in celsius
given a temperature in farenhiet degrees
"""

celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

# layer = tf.keras.layers.Dense(units=1, input_shape=[1])
# model = tf.keras.Sequential([layer])

hidden1 = tf.keras.layers.Dense(units=3, input_shape=[1])
hidden2 = tf.keras.layers.Dense(units=3)
output = tf.keras.layers.Dense(units=1)
model = tf.keras.Sequential([hidden1, hidden2, output])

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

print("Start training...")
history = model.fit(celsius, fahrenheit, epochs=1000, verbose=False)
print("Model trained...")


import matplotlib.pyplot as plt
# plt.xlabel('# Epoch')
# plt.ylabel('Magnitud of loss')
# plt.plot(history.history['loss'])

result = model.predict([100.0])
print('The result is {result} fahrenheit!'.format(result=result))


print('Internal variables of the model')
print(hidden1.get_weights())