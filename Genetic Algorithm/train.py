"""
Utility used by the Network class to actually train.

Based on:
    https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py

"""
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras import backend as K

# Helper: Early stopping.
early_stopper = EarlyStopping(patience=5)

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

def compile_model(network, nb_classes, input_shape):
    """Compile a sequential model.

    Args:
        network (dict): the parameters of the network

    Returns:
        a compiled network.

    """
    # Get our network parameters.
    nb_layers = network['nb_layers']
    activation = network['activation']
    optimizer = network['optimizer']

    model = Sequential()

    # Add each layer.
    for i in range(nb_layers):
        model.add(Conv2D(pow(2, 5+i), (5, 5), padding="same", input_shape=[28, 28, 1], activation=activation))
        model.add(Conv2D(pow(2, 5+i+1), (5, 5), padding="same", input_shape=[28, 28, 1], activation=activation))
        model.add(MaxPool2D((2,2)))

    model.add(Dropout(0.2))  # hard-coded dropout
    model.add(Flatten())
    # Output layer.
    model.add(Dense(nb_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                  metrics=['accuracy'])

    return model

def train_and_score(network):
    """Train the model, return test loss.

    Args:
        network (dict): the parameters of the network
        dataset (str): Dataset to use for training/evaluating

    """
    nb_classes, batch_size, input_shape, x_train, \
            x_test, y_train, y_test = get_fashion_mnist()
    print("Composition: "+str(network))
    model = compile_model(network, nb_classes, input_shape)

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=10000,  # using early stopping, so no real limit
              verbose=0,
              validation_data=(x_test, y_test),
              callbacks=[early_stopper])

    score = model.evaluate(x_test, y_test, verbose=0)

    return score[1]  # 1 is accuracy. 0 is loss.
