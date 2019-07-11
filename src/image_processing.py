from PIL import Image
import glob
import cairosvg
import os
import pickle
'''This needs to be run from the src/ directory to work, if running from other directories, change data_path and pokemon_name accordingly'''
image_dict = {}
data_path = '../data/dataset'
def read_data():
    for directoryname in glob.glob(data_path + '/*'):
        image_list = []
        pokemon_name = directoryname[16:]
        for filename in glob.glob(directoryname + "/*"):
            try:
                im=Image.open(filename)
            except:
                print("Broken stuff? {}".format(filename))
                try:
                    cairosvg.svg2png(url=filename, write_to=filename[:-3]+"png")
                    os.remove(filename[:-3]+"svg")
                except:
                    print("Still broken image: {}".format(filename))
                    try: 
                        im=Image.open(filename[:-3]+"png")
                    except:
                        print("This picture is STILL BROKEN: {}".format(filename))
                image_list.append(im)
                im.close()

        image_dict[pokemon_name] = image_list

def rename_save_all():
    '''Small function to rename all the random file names to ones relevant to the pokemon'''
    for key in image_dict.keys():
        dir_path = data_path + "/" + key
        print("Currently changing: {}".format(key))
        for idx, filename in enumerate(glob.glob(dir_path + "/*")):
            im = Image.open(filename)
            file_path = dir_path + "/" + "{0}_{1}".format(key, idx) + ".jpeg"
            try:
                im.save(file_path, format="jpeg")
            except:
                im = im.convert("RGB")
                im.save(file_path, format="jpeg")
            im.close()

def rename_save_one(pokemon="Abra"):
    '''Test function before accidentally nuking an entire dataset'''
    dir_path = data_path + "/" + pokemon
    for idx, filename in enumerate(glob.glob(dir_path + "/*")):
        im = Image.open(filename)
        file_path = dir_path + "/" + "{0}_{1}".format(pokemon, idx) + ".jpeg"
        try:
            im.save(file_path, format="jpeg")
        except:
            try:
                im = im.convert("RGB")
            except:
                print("This file sucks: {}".format(file_path))
            im.save(file_path, format="jpeg")
        im.close()

def delete_stupid_one(pokemon="Abra"):
    '''deletes the original files as the previous function saves new ones'''
    dir_path = data_path + "/" + pokemon
    for img in os.listdir(dir_path):
        file_cond = img.startswith(pokemon)
        ds_cond = img.startswith(".")
        if not file_cond:
            if not ds_cond:
                os.remove(dir_path+'/'+img)

def delete_stupid_all():
    '''deletes the original files as the previous function saves new ones'''
    for key in image_dict.keys():
        dir_path = data_path + "/" + key
        for img in os.listdir(dir_path):
            file_cond = img.startswith(key)
            ds_cond = img.startswith(".")
            if not file_cond:
                if not ds_cond:
                    os.remove(dir_path+'/'+img)


def oh_shit_go_back():
    '''in case you save too many copies of things and need to go back and delete them'''
    for directoryname in glob.glob(data_path + '/*'):
        for filename in glob.glob(directoryname + "/*"):
            file_cond = directoryname[16:]+"_" in filename
            ds_cond = filename.startswith(".D")
            if file_cond:
                if not ds_cond:
                    os.remove(filename)

def image_count(image_dicitonary=None):
    count_dict = {}
    for key in image_dicitonary.keys():
        count_dict[key] = len(image_dicitonary[key])
    return count_dict

def pickle_var(to_be_pickled=None, file_name="pickle"):
    pickle_dir_name = "../pickles/"
    if to_be_pickled is not None:
        pickle.dump(to_be_pickled, open(pickle_dir_name + file_name + ".p", "wb"))
    else:
        return "Add file to be pickled"