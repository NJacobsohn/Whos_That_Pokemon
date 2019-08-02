"""
THIS CODE IS VERY VERY DEPRECATED, ALL CONTENTS AND FUNCTIONS WERE MOVED INTO cnn_class.py INSTEAD
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, SpatialDropout2D
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization, SeparableConv2D
from keras.layers.convolutional import Conv2D
from keras.utils import np_utils
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import os
import pickle
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

"""
THIS CODE IS VERY VERY DEPRECATED, ALL CONTENTS AND FUNCTIONS WERE MOVED INTO cnn_class.py INSTEAD
"""

def build_cnn(input_shape=(64, 64, 3), nb_classes = 149, neurons = 32, nb_filters = 16, pool_size = (2, 2), kernel_size = (3, 3)):
    """
    Input is desired settings for CNN training
    current defaults are for the data with 1 generation of pokemon images
    """
    model = Sequential() # model is a linear stack of layers (don't change)

    model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]),
                    padding='valid',
                    input_shape=input_shape)) #first conv. layer
    #model.add(BatchNormalization(axis=-1))
    model.add(Activation('tanh'))

    model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]), padding='same')) #2nd conv. layer 
    #model.add(BatchNormalization(axis=-1))
    model.add(Activation('tanh'))

    model.add(MaxPooling2D(pool_size=pool_size)) # decreases size, helps prevent overfitting
    model.add(Dropout(0.25)) # zeros out some fraction of inputs, helps prevent overfitting

    model.add(SeparableConv2D(nb_filters+32, (kernel_size[0], kernel_size[1]), padding='same')) #3rd conv. layer
    #model.add(BatchNormalization(axis=-1))
    model.add(Activation('tanh'))
    model.add(SeparableConv2D(nb_filters+32, (kernel_size[0], kernel_size[1]), padding='same')) #4th conv. layer
    #model.add(BatchNormalization(axis=-1))
    model.add(Activation('tanh'))

    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))

    '''
    model.add(SeparableConv2D(nb_filters+32, (kernel_size[0], kernel_size[1]), padding='same')) #5th conv. layer
    #model.add(BatchNormalization(axis=-1))
    model.add(Activation('tanh'))
    model.add(SeparableConv2D(nb_filters+32, (kernel_size[0], kernel_size[1]), padding='same')) #6th conv. layer
    #model.add(BatchNormalization(axis=-1))
    model.add(Activation('tanh'))
    

    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))
    '''

    model.add(Flatten()) # necessary to flatten before going into conventional dense layer
    print('Model flattened out to ', model.output_shape)

    model.add(Dense(neurons)) # neurons can change
    model.add(Activation('relu'))


    model.add(Dense(nb_classes)) # 149 final nodes (one for each class)
    model.add(Activation('softmax')) # keep softmax at end to pick between classes 0-148

        
    # evaluation
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    
    return model
# optimizers: 'adam', 'adadelta', 'sgd'
# activation functions: 'linear', 'sigmoid', 'tanh', 'relu', 'softplus', 'softsign'
"""
THIS CODE IS VERY VERY DEPRECATED, ALL CONTENTS AND FUNCTIONS WERE MOVED INTO cnn_class.py INSTEAD
"""

def create_data_generators(input_shape=(64, 64), batch_size=64):
    train_datagen = ImageDataGenerator(
     rotation_range=6,
     width_shift_range=0.1,
     height_shift_range=0.1,
     brightness_range=[0.2, 0.8],
     shear_range=0.1,
     zoom_range=0.1,
     horizontal_flip=True
     )


    test_datagen = ImageDataGenerator()


    train_generator = train_datagen.flow_from_directory(
        '../data/train',
        target_size=input_shape,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True)
        
    holdout_generator = test_datagen.flow_from_directory(
        '../data/val',
        target_size=input_shape,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

    test_generator = test_datagen.flow_from_directory(
        '../data/test',
        target_size=input_shape,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

    return train_generator, test_generator, holdout_generator


def predict_classes(model, test_generator, n_test, batch_size, metrics_path):
    Y_pred = model.predict_generator(test_generator, 
                                    steps=n_test/batch_size,
                                    use_multiprocessing=True, 
                                    verbose=1)
    """
    Create confusion matrix and classification report for holdout set
    """
    # Take the predicted label for each observation
    y_pred = np.argmax(Y_pred, axis=1)

    # Create confusion matrix and save it
    cm = confusion_matrix(holdout_generator.classes, y_pred)
    with open(metrics_path + "_cm.txt", 'wb') as f:
        pickle.dump(cm, f)

    # Create classification report and save it
    with open('../pickles/class_names.p', 'rb') as f:
        class_names = np.array(pickle.load(f))

    class_report = classification_report(holdout_generator.classes, y_pred, target_names=class_names)
    print(classification_report)
    with open(metrics_path + "_cr.txt", 'w') as f:
        f.write(repr(class_report))

"""
THIS CODE IS VERY VERY DEPRECATED, ALL CONTENTS AND FUNCTIONS WERE MOVED INTO cnn_class.py INSTEAD
"""



if __name__ == "__main__":

    n_train = sum(len(files) for _, _, files in os.walk("../data/train"))  # : number of training samples

    n_test = sum(len(files) for _, _, files in os.walk("../data/val"))  # : number of validation samples

    n_holdout = sum(len(files) for _, _, files in os.walk("../data/test"))

    in_shape = (64, 64)
    batch = 32
    train_generator, test_generator, holdout_generator = create_data_generators(in_shape, batch)
    
    model = build_cnn(input_shape=(in_shape[0], in_shape[1], 3), nb_classes = 149, neurons = 128, nb_filters = 16, pool_size = (2, 2), kernel_size = (3, 3))


    model.fit_generator(
        train_generator,
        steps_per_epoch=n_train/batch,
        epochs=10,
        verbose=1,
        validation_data=test_generator,
        validation_steps=n_test/batch,#,
        #workers=12,
        use_multiprocessing=True)

    metrics = model.evaluate_generator(holdout_generator,
                                           steps=n_holdout/batch,
                                           use_multiprocessing=True,
                                           verbose=1)
    print(f"Holdout loss: {metrics[0]} Accuracy: {metrics[1]}")
    acc = metrics[1]
    acc = str(acc)[2:6]
    model_path = "../models/model_acc" + acc + ".h5"
    model.save(model_path)
    print("Saved model to \"" + model_path + "\"")
    model_metrics = "../models/metrics/model_acc" + acc

    predict_classes(model, holdout_generator, n_holdout, batch, model_metrics)
    print("Saved CM and Classification Report to \"" + model_metrics + "\"")


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