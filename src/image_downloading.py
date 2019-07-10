from google_images_download import google_images_download

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