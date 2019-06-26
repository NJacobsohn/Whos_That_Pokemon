import split_folders
import glob
'''
Once again, run this from the src folder, this makes train/test/holdout data in respective folders
This code will also (optionally) print out the total directories in the train/test/holdout folders and total files in each subdirectory
'''
data_path = '../data'
folder_ls = ["train/*", "val/*", "test/*"]
def split():
    split_folders.ratio('../data/dataset', output="../data", seed=1337, ratio=(.64, .16, .2)) #code labels train/test/holdout folders as train/val/test

def print_split(verbose=False):
    '''prints the breakdown of files and folders in train/test/holdout'''
    for name in folder_ls:
        train_val_test = 0
        for _ in glob.glob(data_path + "/" + name):
            train_val_test += 1
        print("The {0} folder has {1} sub-folders.".format(name[:-2], train_val_test))
        print("-" * 20)
        if verbose:
            for directoryname in glob.glob(data_path + "/" + name):
                count = 0
                for _ in glob.glob(directoryname + "/*"):
                    count += 1
                print("The sub-folder {0} has {1} images".format(directoryname[-10:], count))

            