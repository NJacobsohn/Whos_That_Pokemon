from google_images_download import google_images_download
from collections import defaultdict
import os
import numpy as np
import pickle
from image_processing import pickle_var as pv
from shutil import copyfile

#This takes a very, very, long time and needs constant internet connection, do NOT start this if you have to go anywhere
#For 10 Pokemon (115 images each) to complete, it takes 15.5 minutes, so ~95 seconds per keyword
#100 pokemon will take 2.5 hours!
#150 pokemon will take 3.75 hours!

#This function was originally planned to download images, but I decided downloading them individually THEN sorting would be better.
"""
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
    paths = gid.download(arguments)
    """

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


def make_master_dict(grouped_pokemon_list):
    master_dict = defaultdict(list)
    for group in grouped_pokemon_list:
        evolution_line = group[-1]
        master_dict[evolution_line] = group
    return master_dict

def make_grouped_data_copy(master_dict, path_to_data='../data/dataset/', path_to_new_data="../data/gen1m/fulldata/"):
    for master_dir, evo_line in master_dict.items():
        print("Current Directory: {}".format(master_dir))
        destination_directory_path = path_to_new_data + master_dir
        os.mkdir(destination_directory_path)
        for pokemon_name in evo_line:
            dir_path = path_to_data + pokemon_name + "/"
            for img in os.listdir(dir_path):
                img_source = dir_path + img
                img_destination = destination_directory_path + "/" + img
                copyfile(img_source, img_destination)



#__________________________ GENERATION 1&2 __________________________#
"""
gen12 = [
['Bulbasaur', 'Ivysaur', 'Venusaur'], ['Charmander', 'Charmeleon', 'Charizard'], ['Squirtle', 'Wartortle', 'Blastoise'], ['Caterpie', 'Metapod', 'Butterfree'], ['Weedle', 'Kakuna', 'Beedrill'],
['Pidgey', 'Pidgeotto', 'Pidgeot'], ['Rattata', 'Raticate'], ['Spearow', 'Fearow'], ['Ekans', 'Arbok'], ['Pichu', 'Pikachu', 'Raichu'], ['Sandshrew', 'Sandslash'],
['Nidorina', 'Nidoqueen'], ['Nidorino', 'Nidoking'], ['Cleffa', 'Clefairy', 'Clefable'], ['Vulpix', 'Ninetales'],
['Igglybuff', 'Jigglypuff', 'Wigglytuff'], ['Zubat', 'Golbat', 'Crobat'],
['Oddish', 'Gloom', 'Vileplume', 'Bellossom'], ['Paras', 'Parasect'], ['Venonat', 'Venomoth'], ['Diglett', 'Dugtrio'], ['Meowth', 'Persian'], ['Psyduck', 'Golduck'], ['Mankey', 'Primeape'],
['Growlithe', 'Arcanine'], ['Poliwag', 'Poliwhirl', 'Poliwrath', 'Politoed'], ['Abra', 'Kadabra', 'Alakazam'], ['Machop', 'Machoke', 'Machamp'], ['Bellsprout', 'Weepinbell', 'Victreebel'],
['Tentacool', 'Tentacruel'], ['Geodude', 'Graveler', 'Golem'], ['Ponyta', 'Rapidash'], ['Slowpoke', 'Slowbro', 'Slowking'], ['Magnemite', 'Magneton'], ["Farfetchd"], ['Doduo', 'Dodrio'],
['Seel', 'Dewgong'], ['Grimer', 'Muk'], ['Shellder', 'Cloyster'], ['Gastly', 'Haunter', 'Gengar'], ['Onix', 'Steelix'], ['Drowzee', 'Hypno'], ['Krabby', 'Kingler'], ['Voltorb', 'Electrode'],
['Exeggcute', 'Exeggutor'], ['Cubone', 'Marowak'], ['Tyrogue', 'Hitmonlee', 'Hitmonchan', 'Hitmontop'], ['Lickitung'], ['Koffing', 'Weezing'], ['Rhyhorn', 'Rhydon'],
['Chansey', 'Blissey'], ['Tangela'], ['Kangaskhan'],
['Horsea', 'Seadra', 'Kingdra'], ['Goldeen', 'Seaking'], ['Staryu', 'Starmie'], ['MrMime'], ['Scyther', 'Scizor'], ['Smoochum', 'Jynx'], ['Elekid', 'Electabuzz'], ['Magby', 'Magmar'],
['Pinsir'], ['Tauros'], ['Magikarp', 'Gyarados'],
['Lapras'], ['Ditto'], ['Vaporeon', 'Jolteon', 'Flareon', 'Espeon', 'Umbreon', 'Eevee'], ['Porygon', 'Porygon2'], ['Omanyte', 'Omastar'], ['Kabuto', 'Kabutops'], ['Aerodactyl'], ['Snorlax'],
['Articuno'], ['Zapdos'], ['Moltres'], ['Dratini', 'Dragonair', 'Dragonite'], ['Mewtwo'], ['Mew'],
['Chikorita', 'Bayleef', 'Meganium'], ['Cyndaquil', 'Quilava', 'Typhlosion'], ['Totodile', 'Croconaw', 'Feraligatr'], ['Sentret', 'Furret'], ['Hoothoot', 'Noctowl'], ['Ledyba', 'Ledian'],
['Spinarak', 'Ariados'], ['Chinchou', 'Lanturn'], ['Togepi', 'Togetic'], ['Natu', 'Xatu'], ['Mareep', 'Flaaffy', 'Ampharos'], ['Marill', 'Azumarill'], ['Sudowoodo'],
['Hoppip', 'Skiploom', 'Jumpluff'], ['Aipom'], ['Sunkern', 'Sunflora'], ['Yanma'], ['Wooper', 'Quagsire'], ['Murkrow'], ['Misdreavus'], ['Unown'],
['Wobbuffet'], ['Girafarig'], ['Pineco', 'Forretress'], ['Dunsparce'], ['Gligar'], ['Snubbull', 'Granbull'], ['Qwilfish'],
['Shuckle'], ['Heracross'], ['Sneasel'], ['Teddiursa', 'Ursaring'], ['Slugma', 'Magcargo'], ['Swinub', 'Piloswine'], ['Corsola'], ['Remoraid', 'Octillery'], ['Delibird'], ['Mantine'], ['Skarmory'],
['Houndour', 'Houndoom'], ['Phanpy', 'Donphan'], ['Stantler'], ['Smeargle'], ['Miltank'],
['Raikou'], ['Entei'], ['Suicune'], ['Larvitar', 'Pupitar', 'Tyranitar'], ['Lugia'], ['Ho-Oh'], ['Celebi']
]
"""