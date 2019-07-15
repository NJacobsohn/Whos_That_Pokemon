from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, SpatialDropout2D, GlobalAveragePooling2D
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization, SeparableConv2D
from keras.layers.convolutional import Conv2D
from keras.applications import Xception
from keras.utils import np_utils
from keras import metrics
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from keras import callbacks
import os
import pickle
import numpy as np
import boto3

s3 = boto3.resource('s3')
s3_client = boto3.client('s3')

class PokemonCNN(object):
    '''
    This class is meant to make testing various things on my CNN easier, should involve less swapping between terminals and development environments
    '''

    def __init__(self, train_path, val_path, test_path, model_name=None, model_type="CNN", weight_path=None, s3_save=False):
        '''
        Input:
            Files paths for your train/val/test files (or train/test/holdout, whatever you want to call it, I'm not your mother)
        '''
        self.weight_path = weight_path
        self.model_name = model_name
        self.model_save_path = "../models/"
        self.metrics_save_path = "../models/metrics/"
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.model_type = model_type
        if s3_save:
            self.BucketName = "capstone2-pokemon-data"
            self.s3_save = True
        else:
            self.s3_save = False
        if self.model_type.lower() != 'xception':
            self.xception = False
            self.cnn_init()
        elif self.model_type.lower() == 'xception':
            self.xception = True
            self.xception_init()
            self.model_name = "xception"
        self.len_init()


    def cnn_init(self):
        self.param_init()
        self.create_generators()
        self.make_callbacks()

    def xception_init(self):
        self.param_init()
        self.create_generators()
        self.make_callbacks()

    def fit(self):
        '''
        Fits built model to given params
        '''
        print(self.model.summary())
        if self.xception:
            self.hist = self.model.fit_generator(
                        self.train_gen,
                        steps_per_epoch=self.n_train,
                        epochs=self.epochs,
                        validation_data=self.val_gen,
                        validation_steps=self.n_val,
                        use_multiprocessing=True,
                        callbacks=self.callbacks)
        else:
            self.hist = self.model.fit_generator(
                self.train_gen,
                steps_per_epoch=self.n_train/self.batch,
                epochs=self.epochs,
                verbose=1,
                validation_data=self.val_gen,
                validation_steps=self.n_val/self.batch,
                use_multiprocessing=True,
                callbacks = self.callbacks)

    def build_model(self, kernel_size=(3, 3), pool_size=(2, 2), dropout_perc=0.25, num_blocks=1, custom_weights=None):
        if self.xception:
            self.build_xception_model()
        else:
            self.build_cnn_model(kernel_size=kernel_size, pool_size=pool_size, dropout_perc=dropout_perc, num_blocks=num_blocks, custom_weights=custom_weights)

    def create_generators(self, augmentation_strength=0.4):
        '''
        Input:
            augmentation_strength: float between 0 and 1 (higher numbers = more augmentation, use higher than default if your model tends to overfit)  
        '''
        if self.xception:
            train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       rotation_range=augmentation_strength,
                                       shear_range=augmentation_strength,
                                       zoom_range=augmentation_strength,
                                       cval=augmentation_strength,
                                       horizontal_flip=True,
                                       vertical_flip=True)

            test_datagen = ImageDataGenerator(rescale=1. / 255)

    
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

        else:
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

    def build_cnn_model(self, kernel_size=(3, 3), pool_size=(2, 2), dropout_perc=0.25, num_blocks=1, custom_weights=None):
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
        activation_func = 'tanh' #LeakyReLU() #usually 'tanh'
        self.model = Sequential() # model is a linear stack of layers (don't change)

        self.model.add(Conv2D(self.nb_filters, (kernel_size[0], kernel_size[1]),
                    padding='valid',
                    input_shape=(self.image_size[0], self.image_size[1], 3), name="conv1_b1")) #first conv. layer
        self.model.add(Activation(activation_func, name="act1_b1"))

        self.model.add(Conv2D(self.nb_filters, (kernel_size[0], kernel_size[1]), padding='same', name="conv2_b1")) #2nd conv. layer 
        self.model.add(Activation(activation_func, name="act2_b1"))

        self.model.add(MaxPooling2D(pool_size=pool_size, name="maxpool_b1")) # decreases size, helps prevent overfitting
        self.model.add(Dropout(dropout_perc, name="dropout_b1")) # zeros out some fraction of inputs, helps prevent overfitting


        for block_num in range(num_blocks):
            filter_augmentation = 32 * (block_num + 1)
            self.model.add(SeparableConv2D(self.nb_filters+filter_augmentation, (kernel_size[0], kernel_size[1]), padding='same', name="sepconv1_b{}".format(block_num +2))) #3rd conv. layer
            self.model.add(Activation(activation_func, name="act1_b{}".format(block_num +2)))
            self.model.add(SeparableConv2D(self.nb_filters+filter_augmentation, (kernel_size[0], kernel_size[1]), padding='same', name="sepconv2_b{}".format(block_num +2))) #4th conv. layer
            self.model.add(Activation(activation_func, name="act2_b{}".format(block_num +2)))
            self.model.add(MaxPooling2D(pool_size=pool_size, name="maxpool_b{}".format(block_num + 2)))
            self.model.add(Dropout(dropout_perc, name="dropout_b{}".format(block_num + 2)))

        self.model.add(Flatten()) # necessary to flatten before going into conventional dense layer
        print('Model flattened out to ', self.model.output_shape)

        self.model.add(Dense(self.neurons, name="dense1_blockfinal")) # neurons can change
        self.model.add(Activation('relu', name="act1_blockfinal"))

        self.model.add(Dense(self.nb_classes, name="dense2_blockfinal")) # 149 final nodes (one for each class)
        self.model.add(Activation('softmax', name="act2_blockfinal")) # keep softmax at end to pick between classes 0-148
        if self.weight_path is not None:
            self.model.load_weights(self.weight_path, by_name=True, skip_mismatch=True)#, by_name=True)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', self.top_3_accuracy, self.top_5_accuracy])

    def top_3_accuracy(self, y_true, y_pred):
        return metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)
    
    def top_5_accuracy(self, y_true, y_pred):
        return metrics.top_k_categorical_accuracy(y_true, y_pred, k=5)

    def build_xception_model(self):
        base_model = Xception(input_shape=(self.image_size[0], self.image_size[1], 3), weights='imagenet', include_top=False)

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        predictions = Dense(self.nb_classes, activation='softmax')(x)
        self.model = Model(base_model.input, predictions)
        for layer in self.model.layers:
            layer.trainable = False
        self.model.compile(optimizer='nadam', loss='categorical_crossentropy', metrics=['accuracy', self.top_3_accuracy, self.top_5_accuracy])

    def len_init(self):
        self.n_train = sum(len(files) for _, _, files in os.walk(self.train_path))  # number of training samples
        self.n_val = sum(len(files) for _, _, files in os.walk(self.val_path))  # number of validation samples
        self.n_test = sum(len(files) for _, _, files in os.walk(self.test_path)) # number of test samples
        self.nb_classes = sum(len(dirnames) for _, dirnames, _ in os.walk(self.train_path))

    def param_init(self, epochs=10, batch_size=32, image_size=(64, 64), base_filters=16, final_layer_neurons=128):
        if self.xception:
            self.image_size = (299, 299)
            self.learning_rate = 1e-4
            self.momentum = 0.9
            self.batch = 16
        else:
            self.batch = batch_size
            self.image_size = image_size
            self.nb_filters = base_filters
            self.neurons = final_layer_neurons
        self.epochs = epochs

    def make_callbacks(self):
        # Initialize tensorboard for monitoring
        tensorboard = callbacks.TensorBoard(
            log_dir="../models/logs/{}".format(self.model_name),
            histogram_freq=0, 
            batch_size=self.batch,
            write_graph=True,
            write_grads=True, 
            update_freq='epoch')

        # Initialize model checkpoint to save best model
        """
        self.savename = '../logs/{0}/{1}_best.hdf5'.format(self.model_name, self.model_name)
        mc = callbacks.ModelCheckpoint(
            self.savename,
            monitor='val_loss', 
            verbose=0, 
            save_best_only=True,
            save_weights_only=False, 
            mode='auto', 
            period=1)
        """
        self.callbacks = [tensorboard]#, mc]

class CNNAnalytics(PokemonCNN):
    '''
    This class is made to inherit a PokemonCNN model and do all evalutaion and metric saving. It also makes the code a bit cleaner for the above class
    '''

    def evaluate_model(self):
        '''
        Evaluates model accuracy on holdout set
        '''
        if self.xception:
            self.metrics = self.model.evaluate_generator(self.test_gen,
                                           steps=self.n_test,
                                           use_multiprocessing=True,
                                           verbose=1)
        else:
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
        if self.s3_save:
            model_path = self.model_name + ".h5"
            self.model.save(model_path)
            s3_client.upload_file(model_path, self.BucketName, model_path)
            print("Saved model to s3 bucket!")
        else:
            model_path = self.model_save_path + self.model_name + ".h5"
            self.model.save(model_path)
            print("Saved model to \"" + model_path + "\"")

    def save_weights(self):
        if self.s3_save:
            weight_path = self.model_name + "_weights.h5"
            self.model.save_weights(weight_path)
            s3_client.upload_file(weight_path, self.BucketName, weight_path)
            print("Saved weights to s3 bucket!")
        else:
            model_path = self.model_save_path + self.model_name + "_weights.h5"
            self.model.save_weights(model_path)
            print("Saved model weights to \"" + model_path + "\"")

    def save_history(self, hist):
        metrics = ["loss", "acc", "top_3_accuracy", "top_5_accuracy",
                   "val_loss", "val_acc", "val_top_3_accuracy", "val_top_5_accuracy"]
        num_metrics = len(metrics)
        numpy_loss_history = np.empty(shape=(num_metrics, self.epochs))
        for idx in range(num_metrics):
            numpy_loss_history[idx] = hist.history[metrics[idx]]

        if self.s3_save:
            hist_path = "{}_history.txt".format(self.model_name)
            np.savetxt(hist_path, numpy_loss_history.T, delimiter=",")
            s3_client.upload_file(hist_path, self.BucketName, hist_path)
            print("Saved history to s3 bucket!")

        else:
            hist_path = "../models/{}_history.txt".format(self.model_name)
            np.savetxt(hist_path, numpy_loss_history.T, delimiter=",")
            print("Saved model history to \"" + hist_path + "\"")
        
    def save_prediction_metrics(self):
        """
        Create confusion matrix and classification report for holdout set
        """
        if self.xception:
            self.Y_pred = self.model.predict_generator(self.test_gen, 
                                    steps=self.n_test,
                                    use_multiprocessing=True, 
                                    verbose=1)
        else:
            self.Y_pred = self.model.predict_generator(self.test_gen, 
                                    steps=self.n_test/self.batch,
                                    use_multiprocessing=True, 
                                    verbose=1)
        # Take the predicted label for each observation
        self.y_pred = np.argmax(self.Y_pred, axis=1)

        # Create confusion matrix and save it
        cm = confusion_matrix(self.test_gen.classes, self.y_pred)
        if self.s3_save:
            cm_name = self.model_name + "_cm.txt"
            with open(cm_name, 'wb') as f:
                pickle.dump(cm, f)
                print("Saved confusion matrix to \"" + cm_name + "\"")
            s3_client.upload_file(cm_name, self.BucketName, cm_name)
            print("Saved CM to s3 bucket!")

            class_names_pick = s3_client.get_object(Bucket=self.BucketName, Key="class_names_gen12_grouped.p")
            class_names = pickle.loads(class_names_pick['Body'].read())

            class_report = classification_report(self.test_gen.classes, self.y_pred, target_names=class_names)
            cr_name = self.model_name + "_cr.txt"
            with open(cr_name, 'wb') as f:
                pickle.dump(class_report, f)
                print("Saved confusion matrix to \"" + cr_name + "\"")
            s3_client.upload_file(cr_name, self.BucketName, cr_name)
            print("Saved CM to s3 bucket!")

        else:
            metric_path_cm = self.metrics_save_path + self.model_name + "_cm.txt"
            with open(metric_path_cm, 'wb') as f:
                pickle.dump(cm, f)
                print("Saved confusion matrix to \"" + metric_path_cm + "\"")

            with open('../pickles/class_names_gen12_grouped.p', 'rb') as f:
                class_names = np.array(pickle.load(f))

            metric_path_cr = self.metrics_save_path + self.model_name + "_cr.txt"
            class_report = classification_report(self.test_gen.classes, self.y_pred, target_names=class_names)
            with open(metric_path_cr, 'w') as f:
                f.write(repr(class_report))
                print("Saved classification report to \"" + metric_path_cr + "\"")

if __name__ == "__main__":
    #FOR LOCAL RUNNING
    """
    train_path = "../data/gen12/train"
    val_path = "../data/gen12/val"
    test_path = "../data/gen12/test"
    weight_path = "../models/cnn_grouped_weights.h5"
    """

    #FOR EC2 AWS RUNNING
    train_path = "gen12/train"
    val_path = "gen12/val"
    test_path = "gen12/test"
    weight_path = "cnn_grouped_weights.h5"

    print("Creating Class")
    my_cnn = CNNAnalytics(
        train_path, 
        val_path, 
        test_path, 
        model_name="cnn_grouped_gen12", 
        model_type="cnn", 
        weight_path=weight_path, 
        s3_save=False)

    print("Initializing Parameters")
    my_cnn.param_init(
        epochs=25, 
        batch_size=16, 
        image_size=(64, 64), 
        base_filters=16, 
        final_layer_neurons=128)

    print("Creating Generators")
    my_cnn.create_generators(augmentation_strength=0.4)

    print("Building Model")
    my_cnn.build_model(
        kernel_size=(3, 3), 
        pool_size=(2, 2), 
        dropout_perc=0.25, 
        num_blocks=1)

    print("Fitting Model")
    my_cnn.fit()
    print("Evaluating Model")
    my_cnn.evaluate_model()
    print("Saving Model Predictions")
    my_cnn.save_prediction_metrics()
    print("Saving History")
    my_cnn.save_history(hist=my_cnn.hist)
    print("Saving Weights")
    my_cnn.save_weights()
    print("Saving Model")
    my_cnn.save_model()
    print("Everything ran without errors!")