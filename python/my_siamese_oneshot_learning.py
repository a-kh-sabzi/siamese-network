
import data_generator
import matplotlib.pyplot as plt
import keras
import numpy as np
import tensorflow as tf
import os
import useful_functions

data_train_path = "E:\Paper research\Deep Learning\Siamese\omniglot-master\python\images_background"
data_eval_path = "E:\Paper research\Deep Learning\Siamese\omniglot-master\python\images_evaluation"
batch_size = 32
image_shape = (105, 105, 1)
n_training_epochs = 200
lr = 0.001
# filepath for saving the training model checkpoints
filepath = ".\my_siamese_checkpoints\siamese_model_{epoch}.h5"
# checkpoints folder address for checking the latest checkpoint
checkpoints_address = ".\my_siamese_checkpoints"
# one shot testing parameters on the end of each epoch during training
n_way = 20
n_tests = 100

samples_ids, sample_list, alphabet_list, character_list, character_range = data_generator.data_id_generator(path = data_train_path)

data_train = data_generator.data_generator(data_train_path, samples_ids, sample_list,
            alphabet_list, character_list,
            character_range, batch_size)

checkpoint_files = os.listdir(checkpoints_address)
if checkpoint_files:
    epoch_number = 0
    for chechpoint_file in checkpoint_files:
        epoch_number_temp = chechpoint_file.partition('_')[2]
        epoch_number_temp = epoch_number_temp.partition('_')[2]
        epoch_number_temp = epoch_number_temp.partition('.')[0]
        epoch_number_temp = int(epoch_number_temp)
        if epoch_number_temp > epoch_number:
            epoch_number = epoch_number_temp
            model_filename = chechpoint_file
    print(checkpoint_files)
    print(epoch_number)
    print(os.path.join(checkpoints_address, model_filename))

    siamese_model = keras.models.load_model(os.path.join(checkpoints_address, model_filename))

else:
    def get_siamese_model(input_shape):
        '''
        model architecture for a siamese neural network using keras
        '''

        # defining inputs
        first_input = keras.layers.Input(shape = input_shape)
        second_input = keras.layers.Input(shape = input_shape)

        # using sequential model to implement the twin sub-networks
        sub_net = keras.models.Sequential()

        sub_net.add(keras.layers.Conv2D(filters=64, kernel_size=(10,10), activation='relu', input_shape=input_shape))
        sub_net.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))
        sub_net.add(keras.layers.Dropout(0.2))

        sub_net.add(keras.layers.Conv2D(filters=128, kernel_size=(7,7), activation='relu'))
        sub_net.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))
        sub_net.add(keras.layers.Dropout(0.2))

        sub_net.add(keras.layers.Conv2D(filters=128, kernel_size=(4,4), activation='relu'))
        sub_net.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))
        sub_net.add(keras.layers.Dropout(0.2))

        sub_net.add(keras.layers.Conv2D(filters=256, kernel_size=(4,4), activation='relu'))
        sub_net.add(keras.layers.Dropout(0.2))

        sub_net.add(keras.layers.Flatten())
        sub_net.add(keras.layers.Dense(units=4096, activation='sigmoid'))

        # implementing the twin sub-networks
        first_sub_network = sub_net(first_input)
        second_sub_network = sub_net(second_input)

        # add a custom layer to compute the absolute difference between the output of the twin sub-networks
        abs_layer = keras.layers.Lambda(lambda inputs:keras.backend.abs(inputs[0] - inputs[1]))
        abs_features = abs_layer([first_sub_network, second_sub_network])

            
        # implementing a dense layer to generate the similarity score
        sim_score = keras.layers.Dense(units=1,activation='sigmoid')(abs_features)
        
        # packaging the model
        siamese_net = keras.models.Model(inputs = [first_input, second_input], outputs = sim_score)
        
        return siamese_net

    epoch_number = 0
    siamese_model = get_siamese_model(input_shape = image_shape)
    siamese_model.summary()

    # saving the model's graph
    keras.utils.plot_model(siamese_model, to_file='model.png', show_shapes=True)

    # learning_rate=0.00006
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    siamese_model.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])

# define the checkpoint to save and resume the training
checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=False, mode='min')
# define custom callback for evaluating the network after each epoch
class one_shot_test_callback(keras.callbacks.Callback):
    def __init__(self, model, path,
            samples_ids, sample_list, alphabet_list, 
            character_list, character_range,
            n_way, n_tests):
        self.model = model
        self.path = path
        self.samples_ids = samples_ids
        self.sample_list = sample_list
        self.alphabet_list = alphabet_list
        self.character_list = character_list
        self.character_range = character_range
        self.n_way = n_way
        self.n_tests = n_tests
    def on_epoch_end(self, epoch, logs=None):
        correct_percent = useful_functions.on_shot_testing(self.model, self.path, 
                                        self.samples_ids, self.sample_list, self.alphabet_list, 
                                        self.character_list, self.character_range, 
                                        self.n_way, self.n_tests)
        print('Accuracy of {}-way one shot testing for {} samples at epoch of {} is {}%.'.format(self.n_way,
                self.n_tests, epoch, correct_percent))

custom_callback = one_shot_test_callback(siamese_model, data_train_path,
                                        samples_ids, sample_list, alphabet_list, 
                                        character_list, character_range,
                                        n_way, n_tests)

callbacks_list = [checkpoint, custom_callback]

siamese_model.fit(x = data_train, epochs = n_training_epochs, callbacks=callbacks_list, initial_epoch = epoch_number)
