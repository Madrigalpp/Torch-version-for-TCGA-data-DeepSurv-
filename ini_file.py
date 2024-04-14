import os
import h5py
import configparser

def get_subdirectories(path):
    subdirectories = [f.path for f in os.scandir(path) if f.is_dir()]
    return subdirectories

def get_files_in_directory(directory):
    files = [f.path for f in os.scandir(directory) if f.is_file()]
    return files


def create_ini_file(file_name, output_ini_path, first_dimension):
    config = configparser.ConfigParser()
    config.add_section('train')
    config.set('train', 'h5_file', f'"{file_name}"')
    config.set('train', 'epochs', '1000')
    config.set('train', 'learning_rate', '3.194e-3')
    config.set('train', 'lr_decay_rate', '3.173e-4')
    config.set('train', 'optimizer', '"Adam"')

    config.add_section('network')
    config.set('network', 'drop', '0.401')
    config.set('network', 'norm', 'True')
    config.set('network', 'dims', f'[{first_dimension}, 128, 64, 1]')
    config.set('network', 'activation', '"Tanh"')
    config.set('network', 'l2_reg', '0')

    with open(output_ini_path, 'w') as configfile:
        config.write(configfile)

your_directory_path = 'data'

subdirectories = get_subdirectories(your_directory_path)

file_data_dict = {}
import os

def remove_ds_store(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file == ".DS_Store":
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"Removed: {file_path}")

project_root = "data"
remove_ds_store(project_root)

for subdir in subdirectories:
    files_in_subdir = get_files_in_directory(subdir)
    for file_path in files_in_subdir:
        file_name = os.path.basename(file_path)
        print(f'  File: "{file_path}"')

        dataset = h5py.File(file_path, 'r')
        dataset = dataset['test']
        e = dataset.get('x')

        file_data_dict[file_path] = e[1].shape

print(file_data_dict)

configs_folder = 'configs-coxnnet'

file_dimensions = file_data_dict

if not os.path.exists(configs_folder):
    os.makedirs(configs_folder)

for file_path, dimensions in file_dimensions.items():
    ini_file_path = os.path.join(configs_folder, os.path.splitext(os.path.basename(file_path))[0] + '.ini')
    create_ini_file(file_path, ini_file_path, dimensions[0])  # Change filename to filepath
