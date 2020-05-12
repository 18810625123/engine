import tensorflow as tf
Dense = tf.keras.layers.Dense
Conv2D = tf.keras.layers.Conv2D
Flatten = tf.keras.layers.Flatten
MaxPooling2D = tf.keras.layers.MaxPooling2D
Dropout = tf.keras.layers.Dropout
Sequential = tf.keras.models.Sequential
train_X, train_y = tf.keras.datasets.mnist.load_data()[0]
train_X = train_X.reshape(-1, 28, 28, 1)
train_X = train_X.astype('float32')
train_X /= 255
train_y = tf.keras.utils.to_categorical(
    train_y, num_classes=10, dtype='float32'
)

model = Sequential()
model.add(Conv2D(32, (5,5), activation='relu', input_shape=[28, 28, 1]))
model.add(Conv2D(64, (5,5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss=tf.keras.losses.categorical_crossentropy,
                optimizer=tf.keras.optimizers.Adadelta(learning_rate=0.001, rho=0.95, epsilon=1e-07, name='Adadelta'),
                metrics=['accuracy'])
# model.summary()
batch_size = 100
epochs = 8
# model.fit(train_X, train_y,
#          batch_size=batch_size,
#          epochs=epochs)

test_X, test_y = tf.keras.datasets.mnist.load_data()[1]
test_X = test_X.reshape(-1, 28, 28, 1)
test_X = test_X.astype('float32')
test_X /= 255
test_y = tf.keras.utils.to_categorical(test_y, 10)

print(type(test_X), test_X.shape)
print(type(test_X[0]), test_X[0].shape)

r = model.predict(test_X)
# loss, accuracy = model.evaluate(test_X, test_y, verbose=1)
# print('loss:%.4f accuracy:%.4f' %(loss, accuracy))