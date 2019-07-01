from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, SpatialDropout2D
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization, SeparableConv2D
from keras.layers.convolutional import Conv2D
from keras.utils import np_utils
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from keras import callbacks
import os
import pickle
import numpy as np

class PokemonCNN(object):
    '''
    This class is meant to make testing various things on my CNN easier, should involve less swapping between terminals and development environments
    '''

    def __init__(self, train_path, val_path, test_path, model_name=None):
        '''
        Input:
            Files paths for your train/val/test files (or train/test/holdout, whatever you want to call it, I'm not your mother)
        '''
        self.n_train = None
        self.n_val = None
        self.n_test = None
        self.model = None
        self.epochs = None
        self.batch = None
        self.model_name = model_name
        self.model_save_path = "../models/"
        self.metrics_save_path = "../models/metrics/"
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.train_gen = None
        self.val_gen = None
        self.test_gen = None
        self.nb_classes = None
        self._len_init()
        self.param_init()
        self.create_generators()
        self.make_callbacks()

    def fit(self):
        '''
        Fits built model to given params
        '''
        if self.train_gen and self.val_gen:
            self.hist = self.model.fit_generator(
                self.train_gen,
                steps_per_epoch=self.n_train/self.batch,
                epochs=self.epochs,
                verbose=1,
                validation_data=self.val_gen,
                validation_steps=self.n_val/self.batch,
                use_multiprocessing=True,
                callbacks = self.callbacks
            )
        else:
            print("No image generators found! (This message shouldn't be seen)")
            self.create_generators()

    def save_prediction_metrics(self):
        """
        Create confusion matrix and classification report for holdout set
        """
        Y_pred = self.model.predict_generator(self.test_gen, 
                                    steps=self.n_test/self.batch,
                                    use_multiprocessing=True, 
                                    verbose=1)
        # Take the predicted label for each observation
        y_pred = np.argmax(Y_pred, axis=1)

        # Create confusion matrix and save it
        cm = confusion_matrix(self.test_gen.classes, y_pred)
        metric_path = self.metrics_save_path + self.model_name
        with open(metric_path + "_cm.txt", 'wb') as f:
            pickle.dump(cm, f)
            print("Saved confusion matrix to \"" + metric_path + "_cm.txt\"")

        # Create classification report and save it
        with open('../pickles/class_names.p', 'rb') as f:
            class_names = np.array(pickle.load(f))

        class_report = classification_report(self.test_gen.classes, y_pred, target_names=class_names)
        with open(metric_path + "_cr.txt", 'w') as f:
            f.write(repr(class_report))
            print("Saved classification report to \"" + metric_path + "_cr.txt\"")

    def create_generators(self, augmentation_strength=0.4):
        '''
        Input:
            augmentation_strength: float between 0 and 1 (higher numbers = more augmentation, use higher than default if your model tends to overfit)  
        '''
        train_datagen = ImageDataGenerator(
            rotation_range=15/augmentation_strength,
            width_shift_range=augmentation_strength/4,
            height_shift_range=augmentation_strength/4,
            brightness_range=[0.2, 0.8],
            shear_range=augmentation_strength/4,
            zoom_range=augmentation_strength/4,
            horizontal_flip=True)

        test_datagen = ImageDataGenerator()

        self.train_gen = train_datagen.flow_from_directory(
            self.train_path,
            target_size=self.image_size,
            batch_size=self.batch,
            class_mode='categorical',
            shuffle=True)
        
        self.val_gen = test_datagen.flow_from_directory(
            self.val_path,
            target_size=self.image_size,
            batch_size=self.batch,
            class_mode='categorical',
            shuffle=False)

        self.test_gen = test_datagen.flow_from_directory(
            self.test_path,
            target_size=self.image_size,
            batch_size=self.batch,
            class_mode='categorical',
            shuffle=False)


    def build_model(self, kernel_size=(3, 3), pool_size=(2, 2), droupout_perc=0.25, num_blocks=1):
        '''
        INPUT:
            kernel_size (tuple): set filter size
            pool_size (tuple): set pooling size
            dropout_perc (float between 0 and 1): percent for dropout layers (try to keep between 0.25 and 0.5)
            num_blocks (int): number of layer blocks added to network
                A layer block is:
                    SeparableConv2D layer with additional filters (self.nb_filters + (32 * block (i.e. first block has 32 extra, second block has 64 extra, etc.)))
                    tanh Activation layer
                    SeparableConv2D layer with additional filters (self.nb_filters + (32 * block (i.e. first block has 32 extra, second block has 64 extra, etc.)))
                    tanh Activation layer
                    MaxPooling2D layer (default to (2, 2) pool size)
                    Dropout layer (default 0.25)
        '''
        self.model = Sequential() # model is a linear stack of layers (don't change)

        self.model.add(Conv2D(self.nb_filters, (kernel_size[0], kernel_size[1]),
                    padding='valid',
                    input_shape=(self.image_size[0], self.image_size[1], 3))) #first conv. layer
        self.model.add(Activation('tanh'))

        self.model.add(Conv2D(self.nb_filters, (kernel_size[0], kernel_size[1]), padding='same')) #2nd conv. layer 
        self.model.add(Activation('tanh'))

        self.model.add(MaxPooling2D(pool_size=pool_size)) # decreases size, helps prevent overfitting
        self.model.add(Dropout(droupout_perc)) # zeros out some fraction of inputs, helps prevent overfitting


        for block_num in range(num_blocks):
            filter_augmentation = 32 * (block_num + 1)
            self.model.add(SeparableConv2D(self.nb_filters+filter_augmentation, (kernel_size[0], kernel_size[1]), padding='same')) #3rd conv. layer
            self.model.add(Activation('tanh'))
            self.model.add(SeparableConv2D(self.nb_filters+filter_augmentation, (kernel_size[0], kernel_size[1]), padding='same')) #4th conv. layer
            self.model.add(Activation('tanh'))
            self.model.add(MaxPooling2D(pool_size=pool_size))
            self.model.add(Dropout(droupout_perc))

        self.model.add(Flatten()) # necessary to flatten before going into conventional dense layer
        print('Model flattened out to ', self.model.output_shape)

        self.model.add(Dense(self.neurons)) # neurons can change
        self.model.add(Activation('relu'))

        self.model.add(Dense(self.nb_classes)) # 149 final nodes (one for each class)
        self.model.add(Activation('softmax')) # keep softmax at end to pick between classes 0-148
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def _len_init(self):
        self.n_train = sum(len(files) for _, _, files in os.walk(self.train_path))  # number of training samples
        self.n_val = sum(len(files) for _, _, files in os.walk(self.val_path))  # number of validation samples
        self.n_test = sum(len(files) for _, _, files in os.walk(self.test_path)) # number of test samples
        self.nb_classes = sum(len(dirnames) for _, dirnames, _ in os.walk(self.train_path))

    def param_init(self, epochs=10, batch_size=32, image_size=(64, 64), base_filters=16, final_layer_neurons=128):
        self.epochs = epochs
        self.batch = batch_size
        self.image_size = image_size
        self.nb_filters = base_filters
        self.neurons = final_layer_neurons

    def evaluate_model(self):
        '''
        Evaluates model accuracy on holdout set
        '''
        self.metrics = self.model.evaluate_generator(self.test_gen,
                                           steps=self.n_test/self.batch,
                                           use_multiprocessing=True,
                                           verbose=1)
        if self.model_name is None:
            acc = self.metrics[1]
            acc = str(acc)[2:6]
            self.model_name = "model_accuracy_" + acc
        print(f"Holdout loss: {self.metrics[0]} Accuracy: {self.metrics[1]}")
        
    
    def save_model(self):
        model_path = self.model_save_path + self.model_name + ".h5"
        self.model.save(model_path)
        print("Saved model to \"" + model_path + "\"")

    def make_callbacks(self):
        # Initialize tensorboard for monitoring
        tensorboard = callbacks.TensorBoard(log_dir="../models/",
                                                  histogram_freq=0, batch_size=self.batch,
                                                  write_graph=True, embeddings_freq=0)

        # Initialize model checkpoint to save best model
        self.savename = '../models/' + 'my_cnn_best' + '.hdf5'
        mc = callbacks.ModelCheckpoint(self.savename,
                                             monitor='val_loss', verbose=0, save_best_only=True,
                                             save_weights_only=False, mode='auto', period=1)
        self.callbacks = [mc, tensorboard]
    
    def save_history(self, hist):
        #val_loss, val_acc, loss, acc
        numpy_loss_history = np.empty(shape=(4, self.epochs))
        metrics = ["loss", "acc", "val_loss", "val_acc"]
        for idx, _ in enumerate(numpy_loss_history):
            numpy_loss_history[idx] = hist.history[metrics[idx]]
        np.savetxt("../models/{}_history.txt".format(self.model_name), numpy_loss_history.T, delimiter=",")
        

if __name__ == "__main__":

    train_path = "../data/train"
    val_path = "../data/val"
    test_path = "../data/test"

    print("Creating Class")
    my_cnn = PokemonCNN(train_path, val_path, test_path)
    print("Initializing Parameters")
    my_cnn.param_init(epochs=10, batch_size=32, image_size=(64, 64), base_filters=16, final_layer_neurons=128)
    print("Creating Generators")
    my_cnn.create_generators(augmentation_strength=0.4)
    print("Building Model")
    my_cnn.build_model(kernel_size=(3, 3), pool_size=(2, 2), droupout_perc=0.25, num_blocks=1)
    print("Fitting Model")
    my_cnn.fit()
    print("Evaluating Model")
    my_cnn.evaluate_model()
    print("Saving Model Predictions")
    my_cnn.save_prediction_metrics()
    print("Saving History")
    my_cnn.save_history(hist=my_cnn.hist)
    print("Saving Model")
    my_cnn.save_model()
    print("Everything ran without errors!")
    