from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.utils import np_utils
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def build_cnn(input_shape=(64, 64, 3), nb_classes = 149, neurons = 32, nb_filters = 16, pool_size = (2, 2), kernel_size = (3, 3)):
    """
    Input is desired settings for CNN training
    current defaults are for the data with 1 generation of pokemon images
    """
    model = Sequential() # model is a linear stack of layers (don't change)

    model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]),
                    padding='valid',
                    input_shape=input_shape)) #first conv. layer
    model.add(Activation('tanh'))

    model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]), padding='same')) #2nd conv. layer 
    model.add(Activation('tanh'))

    model.add(MaxPooling2D(pool_size=pool_size)) # decreases size, helps prevent overfitting
    model.add(Dropout(0.25)) # zeros out some fraction of inputs, helps prevent overfitting

    model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]), padding='same')) #3rd conv. layer
    model.add(Activation('tanh'))
    model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]), padding='same')) #4th conv. layer
    model.add(Activation('tanh'))


    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))

    model.add(Flatten()) # necessary to flatten before going into conventional dense layer
    print('Model flattened out to ', model.output_shape)

    model.add(Dense(neurons)) # neurons can change
    model.add(Activation('tanh'))


    model.add(Dense(nb_classes)) # 149 final nodes (one for each class)
    model.add(Activation('softmax')) # keep softmax at end to pick between classes 0-148

        
    # evaluation
    model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
    
    return model
# optimizers: 'adam', 'adadelta', 'sgd'
# activation functions: 'linear', 'sigmoid', 'tanh', 'relu', 'softplus', 'softsign'

def create_data_generators():
    datagen = ImageDataGenerator()


    train_generator = datagen.flow_from_directory(
        '../data/train',
        target_size=(64, 64),
        batch_size=64,
        class_mode='categorical',
        shuffle=True)
        
    holdout_generator = datagen.flow_from_directory(
        '../data/test',
        target_size=(64, 64),
        batch_size=64,
        class_mode='categorical',
        shuffle=False)

    validate_generator = datagen.flow_from_directory(
        '../data/val',
        target_size=(64, 64),
        batch_size=64,
        class_mode='categorical',
        shuffle=False)

    return train_generator, validate_generator, holdout_generator




if __name__ == "__main__":

    n_train = sum(len(files) for _, _, files in os.walk("../data/train"))  # : number of training samples

    n_test = sum(len(files) for _, _, files in os.walk("../data/val"))  # : number of validation samples

    n_holdout = sum(len(files) for _, _, files in os.walk("../data/test"))

    train_generator, test_generator, holdout_generator = create_data_generators()
    
    model = build_cnn(input_shape=(64, 64, 3), nb_classes = 149, neurons = 32, nb_filters = 16, pool_size = (2, 2), kernel_size = (3, 3))


    model.fit_generator(
        train_generator,
        steps_per_epoch=n_train/64,
        epochs=5,
        verbose=1,
        validation_data=test_generator,
        validation_steps=n_test/64,
        workers=12,
        use_multiprocessing=True)

    metrics = model.evaluate_generator(holdout_generator,
                                           steps=n_holdout/64,
                                           use_multiprocessing=True,
                                           verbose=1)
    print(f"holdout loss: {metrics[0]} accuracy: {metrics[1]}")
    #score = model.evaluate_generator(test_generator)
    #print('Test score:', score[0])
    #print('Test accuracy:', score[1])
    # during fit process watch train and test error simultaneously
    '''
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))

    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1]) # this is the one we care about
    '''