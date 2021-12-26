
import data_generator
import useful_functions
import os
import keras

data_train_path = r"E:\Paper research\Deep Learning\Siamese\omniglot-master\python\images_background"
data_eval_path = r"E:\Paper research\Deep Learning\Siamese\omniglot-master\python\images_evaluation"
path_list = [data_train_path, data_eval_path]
checkpoints_address_current = r".\my_siamese_checkpoints"
checkpoints_address_backup = r".\back_up_checkpoints\bs_32_lr_00006_with_drop"
checkpoints_list = [checkpoints_address_current, checkpoints_address_backup]
# checkpoints_list = [checkpoints_address_backup]
n_way = 20
n_tests = 500

for checkpoints_address in checkpoints_list:
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

        for path_item in path_list:
            samples_ids, sample_list, alphabet_list, character_list, character_range = data_generator.data_id_generator(path_item)

            correct_percent = useful_functions.on_shot_testing(siamese_model, path_item, 
                                            samples_ids, sample_list, alphabet_list, 
                                            character_list, character_range, 
                                            n_way, n_tests)
            output_str = 'Accuracy of {}-way one shot testing for {} samples from images in:\n{}\nand the model in:\n{}\nis {}%.\n'.format(
                    n_way, n_tests, path_item, checkpoints_address + '\\' + model_filename, correct_percent)
            print(output_str)
            with open('accuracy.txt', 'a') as f:
                f.write(output_str)
    else:
        print('There is no keras model in {}'.format(checkpoints_address))