from google_images_download import google_images_download
from collections import defaultdict
import os
import numpy as np

#This takes a very, very, long time and needs constant internet connection, do NOT start this if you have to go anywhere
#For 10 Pokemon (115 images each) to complete, it takes 15.5 minutes, so ~95 seconds per keyword
#100 pokemon will take 2.5 hours!
#150 pokemon will take 3.75 hours!


def download_keyword(keyword=None, number_of_images=115, output_directory="../data/data_sorted_by_evo/"):
    gid = google_images_download.googleimagesdownload()
    image_directory = "../data/dataset/" + keyword + "/"
    if not keyword:
        return "Please choose a keyword"
    arguments = {"keywords" : keyword,
            "limit" : number_of_images,
            "output_directory" : output_directory,
            "image_directory" : image_directory,
            "print_urls" : True,
            "chromedriver" : "./chromedriver",
            "prefix" : "dl_",
            "no_numbering" : True}   
    gid.download(arguments)

def choose_generation(generation=0):
    pokemon_evo_dictionary = defaultdict(list)
    if generation == -1:
        with open("../data/text_files/all_pokemon_names.txt", "rb") as f:
            names = np.loadtxt(f, dtype=str, delimiter="\n")
    elif generation == 0:
        return "Pick a generation number (-1 for all)"
    else:
        file_path = "../data/text_files/pokemon_gen" + str(generation) + ".txt"
        with open(file_path, "rb") as f:
            names = np.loadtxt(f, dtype=str, delimiter="\n")
    for pokemon in names:
        pokemon_evo_dictionary[pokemon] = []
    return pokemon_evo_dictionary