import tensorflow as tf

from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras import backend as K

def get_fashion_mnist():
    # Set defaults.
    nb_classes = 10
    batch_size = 128
    input_shape = (784,)
    
    img_rows, img_cols = 28,28

    # Get the data.
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    if K.image_data_format() == 'channels_first':
      x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
      x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
      image_shape = (img_rows, img_cols, 1)
    else:
      x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
      x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
      image_shape = (1, img_rows, img_cols)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, nb_classes)
    y_test = to_categorical(y_test, nb_classes)

    return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test)

# Load the fashion-mnist pre-shuffled train data and test data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)
          
nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test = get_fashion_mnist()
            
model = Sequential()

model.add(Conv2D(16, (5, 5), padding="same", input_shape=[28, 28, 1], activation='relu'))
model.add(Conv2D(32, (5, 5), padding="same", input_shape=[28, 28, 1], activation='relu'))
model.add(MaxPool2D((2,2)))
model.add(Dropout(0.2))  # hard-coded dropout

model.add(Conv2D(64, (5, 5), padding="same", input_shape=[28, 28, 1], activation='relu'))
model.add(Conv2D(128, (5, 5), padding="same", input_shape=[28, 28, 1], activation='relu'))
model.add(MaxPool2D((2,2)))
model.add(Dropout(0.2))  # hard-coded dropout



model.add(Dropout(0.2))  # hard-coded dropout
model.add(Flatten())

model.add(Dense(10, activation='softmax'))
# Take a look at the model summary
model.summary()

# checkpoint
filepath="checkpoints/weights.best-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]


model.compile(loss='categorical_crossentropy',
             optimizer='adagrad',
             metrics=['accuracy'])

model.fit(x_train,
         y_train,
         batch_size=64,
         epochs=10000,
         validation_data=(x_test, y_test),
         callbacks=callbacks_list)

# Evaluate the model on test set
score = model.evaluate(x_test, y_test, verbose=0)

# Print test accuracy
print('\n', 'Test accuracy:', score[1])