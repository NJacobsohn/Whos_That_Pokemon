from google_images_download import google_images_download

def download_keywords_from_file(path_to_keywords="../data/download_text.txt", number_of_images=115, output_directory="../data/dataset/"):
    gid = google_images_download.googleimagesdownload()   #class instantiation

    arguments = {"keywords_from_file" : path_to_keywords,
            "limit" : number_of_images,
            "output_directory" : output_directory,
            "print_urls" : True,
            "chromedriver" : "./chromedriver",
            "prefix" : "dl_",
            "no_numbering" : True}   #creating list of arguments
    paths = gid.download(arguments)   #passing the arguments to the function
#print(paths)   #printing absolute paths of the downloaded images