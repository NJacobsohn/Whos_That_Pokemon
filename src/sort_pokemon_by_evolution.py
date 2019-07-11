from google_images_download import google_images_download
from collections import defaultdict
import os
import numpy as np
import pickle
from image_processing import pickle_var as pv

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


#___________________________ GENERATION 1 ___________________________#
"""
gen1 = [['Bulbasaur', 'Ivysaur', 'Venusaur'], ['Charmander', 'Charmeleon', 'Charizard'], ['Squirtle', 'Wartortle', 'Blastoise'], ['Caterpie', 'Metapod', 'Butterfree'], ['Weedle', 'Kakuna', 'Beedrill'],
 ['Pidgey', 'Pidgeotto', 'Pidgeot'], ['Rattata', 'Raticate'], ['Spearow', 'Fearow'], ['Ekans', 'Arbok'], ['Pikachu', 'Raichu'], ['Sandshrew', 'Sandslash'],
 ['Nidoran Female', 'Nidorina', 'Nidoqueen'], ['Nidoran Male', 'Nidorino', 'Nidoking'], ['Clefairy', 'Clefable'], ['Vulpix', 'Ninetales'], ['Jigglypuff', 'Wigglytuff'], ['Zubat', 'Golbat'],
 ['Oddish', 'Gloom', 'Vileplume'], ['Paras', 'Parasect'], ['Venonat', 'Venomoth'], ['Diglett', 'Dugtrio'], ['Meowth', 'Persian'], ['Psyduck', 'Golduck'], ['Mankey', 'Primeape'],
 ['Growlithe', 'Arcanine'], ['Poliwag', 'Poliwhirl', 'Poliwrath'], ['Abra', 'Kadabra', 'Alakazam'], ['Machop', 'Machoke', 'Machamp'], ['Bellsprout', 'Weepinbell', 'Victreebel'],
 ['Tentacool', 'Tentacruel'], ['Geodude', 'Graveler', 'Golem'], ['Ponyta', 'Rapidash'], ['Slowpoke', 'Slowbro'], ['Magnemite', 'Magneton'], ["Farfetch'd"], ['Doduo', 'Dodrio'],
 ['Seel', 'Dewgong'], ['Grimer', 'Muk'], ['Shellder', 'Cloyster'], ['Gastly', 'Haunter', 'Gengar'], ['Onix'], ['Drowzee', 'Hypno'], ['Krabby', 'Kingler'], ['Voltorb', 'Electrode'],
 ['Exeggcute', 'Exeggutor'], ['Cubone', 'Marowak'], ['Hitmonlee', 'Hitmonchan'], ['Lickitung'], ['Koffing', 'Weezing'], ['Rhyhorn', 'Rhydon'], ['Chansey'], ['Tangela'], ['Kangaskhan'],
 ['Horsea', 'Seadra'], ['Goldeen', 'Seaking'], ['Staryu', 'Starmie'], ['Mr. Mime'], ['Scyther'], ['Jynx'], ['Electabuzz'], ['Magmar'], ['Pinsir'], ['Tauros'], ['Magikarp', 'Gyarados'], ['Lapras'],
 ['Ditto'], ['Eevee', 'Vaporeon', 'Jolteon', 'Flareon'], ['Porygon'], ['Omanyte', 'Omastar'], ['Kabuto', 'Kabutops'], ['Aerodactyl'], ['Snorlax'], ['Articuno'], ['Zapdos'], ['Moltres'],
 ['Dratini', 'Dragonair', 'Dragonite'], ['Mewtwo'], ['Mew']]
"""

#___________________________ GENERATION 2 ___________________________#