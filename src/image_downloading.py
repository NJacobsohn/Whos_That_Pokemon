from google_images_download import google_images_download

#This takes a very, very, long time and needs constant internet connection, do NOT start this if you have to go anywhere
#For 10 Pokemon (115 images each) to complete, it takes 15.5 minutes, so ~95 seconds per keyword
#100 pokemon will take 2.5 hours!
#150 pokemon will take 3.75 hours!


def download_keywords_from_file(path_to_keywords="../data/download_text.txt", number_of_images=115, output_directory="../data/dataset/"):
    gid = google_images_download.googleimagesdownload()

    arguments = {"keywords_from_file" : path_to_keywords,
            "limit" : number_of_images,
            "output_directory" : output_directory,
            "print_urls" : True,
            "chromedriver" : "./chromedriver",
            "prefix" : "dl_",
            "no_numbering" : True}   
    paths = gid.download(arguments)