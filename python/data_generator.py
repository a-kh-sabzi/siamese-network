
data_train_path = "E:\Paper research\Deep Learning\Siamese\omniglot-master\python\images_background"
data_eval_path ="E:\Paper research\Deep Learning\Siamese\omniglot-master\python\images_evaluation"

import keras
import math
import os
import useful_functions
import numpy as np

class data_generator(keras.utils.Sequence):

    ''' 
    generates batches of data for keras 
    inputs:
    path --> data path directory
    samples_ids, sample_list, alphabet_list, character_list, character_range --> output of data_id_generator function
    batch_size
    outputs:
    (samples, targets) --> pair of (batch_size, h, w, n_channel), (batch_size, 1)
    '''
    # lines with the "on_epoch_end" comment before them are needed
    # if we want to shuffle our indexes after each epoch 

    def __init__(self, path, 
                samples_ids, sample_list,
                alphabet_list, character_list,
                character_range, batch_size=32,
                # on_epoch_end
                # shuffle=True,
                ):
        self.path = path
        self.samples_ids = samples_ids
        self.n_samples = len(samples_ids)
        self.sample_list = sample_list
        self.alphabet_list = alphabet_list
        self.character_list = character_list
        self.character_range = character_range
        self.n_classes = len(character_list)
        self.batch_size = batch_size

        # self.labels = labels
        # self.dim = dim
        # self.n_channels = n_channels
        # self.n_classes = n_classes

        # on_epoch_end (if you are using shuffle then comment the self.indexes line)
        # self.shuffle = shuffle
        # self.on_epoch_end()
        self.indexes = np.arange(self.n_samples)

        self.samples_ids_np = np.stack(self.samples_ids)

    def __len__(self):
        '''
        Denotes the number of batches per epoch
        each sample is paired with a positive and negative sample
        thus number of batches is twice the n_samples / batch_size
        '''
        return math.floor(2 * self.n_samples / self.batch_size)

    def __getitem__(self, index):
        '''
        Generates one batch of the data.
        In a siamese network, our input is a pair
        half of the pairs are from the same class and the rest  
        '''
        
        # generate the indexes for the current batch (for the used samples in the current batch)
        indexes = self.indexes[(index*self.batch_size)//2:((index+1)*self.batch_size)//2]

        # for each sample, we must find
        # a pair from the same class
        # and a pair from another class 

        batch_ids_main = []
        batch_ids_first_pair = []
        batch_ids_second_pair = []
        for index in indexes:
            main_sample_ids = self.samples_ids_np[index]
            batch_ids_main.append(main_sample_ids)

            main_character = main_sample_ids[2]
            low, high = self.character_range[main_character]
            main_character_samples = self.samples_ids_np[low:high + 1]
            del_id = main_sample_ids[0] - low
            main_character_samples = np.delete(main_character_samples, del_id, 0)
            first_pair_id = np.random.choice(main_character_samples[:,0], size=None, replace=True)
            first_pair_ids = self.samples_ids_np[first_pair_id]
            batch_ids_first_pair.append(first_pair_ids)
            
            character_list_temp = [i for i in range(self.n_classes)]
            character_list_temp = character_list_temp[:main_character] + character_list_temp[main_character + 1:]
            second_character = np.random.choice(character_list_temp, size=None, replace=True)
            low, high = self.character_range[second_character]
            second_character_samples = self.samples_ids_np[low:high + 1]
            second_pair_id = np.random.choice(second_character_samples[:,0], size=None, replace=True)
            second_pair_ids = self.samples_ids_np[second_pair_id]
            batch_ids_second_pair.append(second_pair_ids)


        main_batch, targets = load_data(batch_ids_main, self.path, self.sample_list, self.alphabet_list, self.character_list)
        first_pair_batch, targets = load_data(batch_ids_first_pair, self.path, self.sample_list, self.alphabet_list, self.character_list)
        second_pair_batch, targets = load_data(batch_ids_second_pair, self.path, self.sample_list, self.alphabet_list, self.character_list)

        size, h, w, n_channel = main_batch.shape
        # initialize 2 empty arrays for the input image batch
        pairs=[np.zeros((self.batch_size, h, w, n_channel)) for i in range(2)]   
        # initialize vector for the targets
        targets=np.zeros((self.batch_size,))

        for i in range(self.batch_size):
            pairs[0][i,:,:,:] = main_batch[i//2]
            if i % 2 == 0 :
                pairs[1][i,:,:,:] = first_pair_batch[i//2]
                targets[i] = 1
            else:
                pairs[1][i,:,:,:] = second_pair_batch[i//2]
                targets[i] = 0

        return pairs, targets

    # on_epoch_end
    # def on_epoch_end(self):
    #     '''
    #     Updates indexes after each epoch
    #     '''
    #     self.indexes = np.arange(self.n_samples)
    #     if self.shuffle == True:
    #         np.random.shuffle(self.indexes)

def load_data(batch_ids, path, sample_list, alphabet_list, character_list):
    '''
    takes samples_ids of one batch and their address and returns two numpy arrays.
    outputs:
    batch --> (batch_size, h, w, n_channel) array of data images
    targets --> (batch_size, 1) array of class labels
    path, sample_list, alphabet_list and, character_list are needed to find the images psths
    path is the folder address, alphabet_list and and character_list are the names of sub folders
    and, sample_list is the filenames of images.  
    '''

    batch = []
    targets = []
    for (id_sample, id_alphabet, id_character) in batch_ids:
        sample = sample_list[id_sample]
        alphabet = alphabet_list[id_alphabet]
        character = character_list[id_character]
        image_path = os.path.join(path, alphabet, character, sample)
        img = useful_functions.load_image(image_path, color = 'False')
        if img.ndim == 2:
            img = img.reshape(img.shape[0], img.shape[1], 1)
        batch.append(img)
        targets.append(id_character)
    batch = np.stack(batch)
    targets = np.vstack(targets)

    return batch, targets

def data_id_generator(path):
    '''
    takes the root path of images and returns id of images
    outputs:
    samples_ids --> [(id_sample, id_alphabet, id_character)]
    sample_list --> [sample(image name)]
    alphabet_list --> [alphabet]
    character_list --> [character]
    character_range --> [(character_low, character_high)] sample ids of first and last sample of a character
    '''
    samples_ids = []
    sample_list = []
    alphabet_list = []
    character_list = []
    character_range = []
    i = 0; j = 0; k = 0
    for alphabet in os.listdir(path):
        alphabet_list.append(alphabet)
        alphabet_path = os.path.join(path,alphabet)
        for character in os.listdir(alphabet_path):
            character_list.append(character)
            character_low = k
            character_path = os.path.join(alphabet_path,character)
            for sample in os.listdir(character_path):
                sample_list.append(sample)
                samples_ids.append((k, i, j))
                character_high = k
                k = k + 1
            character_range.append((character_low, character_high))
            j = j + 1
        i = i + 1

    return samples_ids, sample_list, alphabet_list, character_list, character_range

def data_id_random_subset(samples_ids, n_samples):
    '''
    takes list of samples_ids generated by data_id_generator function and randomly chooses n_samples of them
    output:
    samples_ids_subset --> a list of tuples (a subset of samples_ids)
    '''
    id_subset = np.random.choice(len(samples_ids), size=n_samples, replace=False)
    samples_ids_subset = [samples_ids[i] for i in id_subset]
    return samples_ids_subset

class one_shot_data_generator(keras.utils.Sequence):

    ''' 
    generates batches of data for keras predict method
    these batches are prepared for one shot testing of a siamese network
    one shot n_way testing where n is the batch_size 
    inputs:
    path --> data path directory
    samples_ids_subset --> output of data_id_random_subset function (or a subset of samples_ids)
    samples_ids, sample_list, alphabet_list, character_list, character_range --> output of data_id_generator function
    batch_size
    outputs:
    samples --> pair of (batch_size, h, w, n_channel)
    '''
    # lines with the "on_epoch_end" comment before them are needed
    # if we want to shuffle our indexes after each epoch 

    def __init__(self, path,
                samples_ids_subset, 
                samples_ids, sample_list,
                alphabet_list, character_list,
                character_range, batch_size=20,
                # on_epoch_end
                # shuffle=True,
                ):
        self.path = path
        self.samples_ids_subset = samples_ids_subset
        self.samples_ids = samples_ids
        self.n_samples = len(samples_ids_subset)
        self.sample_list = sample_list
        self.alphabet_list = alphabet_list
        self.character_list = character_list
        self.character_range = character_range
        self.n_classes = len(character_list)
        self.batch_size = batch_size

        # self.labels = labels
        # self.dim = dim
        # self.n_channels = n_channels
        # self.n_classes = n_classes

        # on_epoch_end (if you are using shuffle then comment the self.indexes line)
        # self.shuffle = shuffle
        # self.on_epoch_end()
        self.indexes = np.arange(self.n_samples)

        self.samples_ids_np = np.stack(self.samples_ids)
        self.samples_ids_subset = np.stack(self.samples_ids_subset)

    def __len__(self):
        '''
        Denotes the number of batches per epoch
        we do a n-way testing for every sample and
        n is defined equal to the batch_size
        thus the number of batches is n_samples
        '''
        return self.n_samples

    def __getitem__(self, index):
        '''
        Generates one batch of the data.
        In a siamese network, our input is a pair
        this data is for n-way one shot testing
        thus every sample is paired with one random sample from its class
        and n-1 sample from other classes 
        '''
        
        # generate the indexes for the current batch (for the used samples in the current batch)
        # only one non-random sample is used per batch

        # for each sample, we must find
        # a pair from the same class
        # and n-1 pairs from other classes

        batch_ids_main = []
        batch_ids_first_pair = []
        batch_ids_second_pair = []

        main_sample_ids = self.samples_ids_subset[index]
        batch_ids_main.append(main_sample_ids)

        main_character = main_sample_ids[2]
        low, high = self.character_range[main_character]
        main_character_samples = self.samples_ids_np[low:high + 1]
        del_id = main_sample_ids[0] - low
        main_character_samples = np.delete(main_character_samples, del_id, 0)
        first_pair_id = np.random.choice(main_character_samples[:,0], size=None, replace=False)
        first_pair_ids = self.samples_ids_np[first_pair_id]
        batch_ids_first_pair.append(first_pair_ids)
        
        character_list_temp = [i for i in range(self.n_classes)]
        character_list_temp = character_list_temp[:main_character] + character_list_temp[main_character + 1:]
        for second_pair in range(self.batch_size - 1):
            second_character = np.random.choice(character_list_temp, size=None, replace=False)
            low, high = self.character_range[second_character]
            second_character_samples = self.samples_ids_np[low:high + 1]
            second_pair_id = np.random.choice(second_character_samples[:,0], size=None, replace=False)
            second_pair_ids = self.samples_ids_np[second_pair_id]
            batch_ids_second_pair.append(second_pair_ids)


        main_batch, targets = load_data(batch_ids_main, self.path, self.sample_list, self.alphabet_list, self.character_list)
        first_pair_batch, targets = load_data(batch_ids_first_pair, self.path, self.sample_list, self.alphabet_list, self.character_list)
        second_pair_batch, targets = load_data(batch_ids_second_pair, self.path, self.sample_list, self.alphabet_list, self.character_list)

        size, h, w, n_channel = main_batch.shape
        # initialize 2 empty arrays for the input image batch
        pairs=[np.zeros((self.batch_size, h, w, n_channel)) for i in range(2)]   
        # initialize vector for the targets
        # targets aren't needed for keras predict method
        # targets=np.zeros((self.batch_size,))

        for i in range(self.batch_size):
            pairs[0][i,:,:,:] = main_batch[0]
            if i ==  0 :
                pairs[1][i,:,:,:] = first_pair_batch[0]
                # targets[i] = 1
            else:
                pairs[1][i,:,:,:] = second_pair_batch[i - 1]
                # targets[i] = 0

        # return pairs, targets
        return pairs

    # on_epoch_end
    # def on_epoch_end(self):
    #     '''
    #     Updates indexes after each epoch
    #     '''
    #     self.indexes = np.arange(self.n_samples)
    #     if self.shuffle == True:
    #         np.random.shuffle(self.indexes)