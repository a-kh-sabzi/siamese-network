
import cv2
import numpy as np
import data_generator

def load_image(image_path, dim = 1, scale_percent = 100, color = 'False'):
    '''
    takes the image_path and returns a normalized image between 0 and 1
    by default image dimensions are unchanged
    inputs:
    image_path --> string
    dim --> (height, width)
    scale_percent --> percentage of the original size or percentage of the given dim
    color --> 'False' or 'True'
    output:
    img --> image
    '''
    if color == 'False':
        color = cv2.IMREAD_GRAYSCALE
    elif color == 'True':
        color = cv2.IMREAD_COLOR
    else:
        print('Error: Undefined parameter color')
        return 
        
    img = cv2.imread(image_path, color)
    img = img / 255

    if dim == 1:
        dim = (img.shape[0], img.shape[1])
    width = int(dim[1] * scale_percent / 100)
    height = int(dim[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return img

def on_shot_testing(model, path, 
                    samples_ids, sample_list,
                    alphabet_list, character_list, character_range,
                    n_way, n_tests = 100,
                    all_samples = 0,
                    ):
    '''
    takes a siamese keras model and evaluates its n_way on shot accuracy over n_tests
    inputs:
    model --> a keras model 
    path --> address of the root folder of the data
    samples_ids, sample_list, alphabet_list, character_list, character_range --> output of data_id_generator function
    n_way --> n-way one shot testing and here also batch size of our data generator
    n_tests --> number of testings (by default it is 100)
    all_samples --> if it isn't set to 0, testing is done over all samples (by default it is 0)
    outputs:
    correct_percent --> percentage of correct tests (accuracy of our model)
    '''

    if all_samples == 0:
        samples_ids_subset = data_generator.data_id_random_subset(samples_ids, n_tests)
    else:
        samples_ids_subset = samples_ids

    data_test = data_generator.one_shot_data_generator(path, samples_ids_subset,
                samples_ids, sample_list,
                alphabet_list, character_list,
                character_range, batch_size = n_way)

    n_correct = 0
    probs = model.predict(data_test)
    probs = probs.reshape(n_tests,-1)
    print(probs[0])
    for i in range(n_tests):
        if np.argmax(probs[i]) == 0:
            n_correct = n_correct + 1
    correct_percent = (100.0 * n_correct / n_tests)

    return correct_percent