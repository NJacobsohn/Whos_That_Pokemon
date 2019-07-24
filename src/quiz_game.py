from mttkinter import mtTkinter as tk
import PIL
import numpy as np
import pickle
import os
from collections import defaultdict
from PIL import Image, ImageTk

#______________________________________To-Do______________________________________#
"""
Add labels for answer vs. guessed images
Have score constantly showing after user is chosen
Show question progress
Use math to put images in their proper spots each time (regardless of window size)
Show larger image during guessing phase
Make everything much prettier:
    - Background for application
    - Custom borders?
    - Window customization
    - General overlay items?
Add option for # of questions
Add check to make sure you don't get duplicate questions (maybe just recursive function? eg. while pokemon in prev_list)
Show leaderboard on start and finish of quiz
Add option to view advanced stats about your user (# of games, mean score, # of perfect games, etc.)
Update game score saving metrics to include the the questions and guesses for each user rather than flat score

Competitive mode for neural network guesses
Difficulty options against neural net (beating top 1, 3, or 5 accuracy)
"""
#_________________________________________________________________________________#

class QuizGame(tk.Tk):
    """
    This is framework for a singleplayer version of the Pokemon quiz game. This is working on functionality for just one played guessing random images
    Network functionality will be added when the game starts working.
    """
    def __init__(self, master=None):
        tk.Tk.__init__(self)
        self.canvas = tk.Canvas(self, width=1280, height=768)
        self.canvas.pack(side="top", fill="both", expand=True)
        self.img_size = 128, 128
        self.create_widgets()

        #load user scores
        with open("../pickles/user_score_dictionary.p", "rb") as f:
            self.user_dict = pickle.load(f)
        
        #load pokemon and dex number dicts
        with open("../pickles/pokemon_to_dex.p", "rb") as f:
            self.poke_to_dex = pickle.load(f)
        with open("../pickles/dex_to_pokemon.p", "rb") as f:
            self.dex_to_poke = pickle.load(f)
        
        self.score = 0
        self.previous_pokemon = []
        self.previous_guesses = []
        self.is_userCreated = False
        self.question_number = 0


    def create_widgets(self):
        self.username = tk.StringVar()
        self.username.set("Enter Your Username Here")
        self.user_entry = tk.Entry(self, textvariable=self.username)
        self.user_entry.pack(side="top")
        self.user_entry.bind('<Key-Return>', self.create_user)
        self.user_entry.bind('<Key-Return>', self.pick_pokemon_with_image, '+')
        self.user_entry.bind('<Key-Return>', self.display_image, '+')
        self.user_entry.bind('<Key-Return>', self.replace_user_with_guess, '+')

        self.quit_button = tk.Button(self, text="QUIT", fg="red", command=self.quit)
        self.quit_button.pack(side="bottom")


    def replace_user_with_guess(self, event):
        self.user_entry["state"] = "disabled"
        self.guess = tk.StringVar()
        self.guess.set("Who's This Pokemon?")
        self.guess_box = tk.Entry(self, textvariable=self.guess)
        self.guess_box.pack(side="bottom")
        self.guess_box.bind("<Key-Return>", self.update_score)
        self.guess_box.bind("<Key-Return>", self.show_guess_with_answer, "+")
        self.guess_box.bind("<Key-Return>", self.create_next_button, "+")

    def create_next_button(self, event):
        #method to make button that resets the current guess and chosen pokemon
        #then picks new pokemon and asks user for guess again
        self.next_question = tk.Button(self,
            text="Next Question",
            command=self.reset_question)
        self.next_question.pack(side="right")

    def reset_question(self):
        #this method should remove all images being shown on the canvas
        if self.question_number < 5:
            self.guess.set("Who's This Pokemon?")
            self.canvas.delete("answer")
            self.canvas.delete("guess")
            self.next_question.pack_forget()
            self.pick_pokemon_with_image("")
            self.display_image("")
        else:
            pass
            #prepare for closing quiz out
            #self.display_final_score_screen()

    def display_final_score_screen(self):
        self.canvas.delete("answer")
        self.canvas.delete("guess")
        self.next_question.pack_forget()
        self.guess_box.pack_forget()
        """
        for idx, item in enumerate(self.previous_pokemon):
            self.img_file_path = "../data/quiz_photos/{}.jpeg".format(item)
            self.im_f = Image.open(self.img_file_path)
            self.im_f.thumbnail(self.img_size)
            self.tk_im_f = ImageTk.PhotoImage(self.im_f)
            tag_name = "question{}".format(idx)
            self.canvas.create_image(128*idx, 10, anchor="nw", image=self.tk_im_f, tag=tag_name)
        """


    def show_guess_with_answer(self, event):
        self.get_fetch_guess()
        self.canvas.delete("question")
        self.canvas.create_image(100, 0, anchor="nw", image=self.tk_im, tag="answer")
        self.canvas.create_image(500, 0, anchor="nw", image=self.guess_tk_im, tag="guess")

    def pick_pokemon_with_image(self, event):
        self.choose_random_pokemon()
        self.get_pokemon_from_number()

    def choose_random_pokemon(self):
        self.pokemon_choice = np.random.randint(1, 152, size=1)[0]

    def get_pokemon_from_number(self):
        self.pokemon = self.dex_to_poke[self.pokemon_choice]
        self.img_file_path = "../data/quiz_photos/{}.jpeg".format(self.pokemon)
        self.previous_pokemon.append(self.pokemon)
        self.fetch_image("")

    def get_fetch_guess(self):
        self.guess_file_path = "../data/quiz_photos/{}.jpeg".format(
            self.guess.get().lower().capitalize()
            )
        try:
            self.guess_im = Image.open(self.guess_file_path)
            self.guess_im.thumbnail(self.img_size)
            self.guess_tk_im = ImageTk.PhotoImage(self.guess_im)
        except:
            print("That's not a Pokemon!")


    def quit(self):
        self.save_user_score()
        self.save_scores()
        self.canvas.destroy()

    def update_score(self, event):
        if self.check_answer(""):
            self.score += 1
            self.answer = True
            self.question_number += 1
        else:
            self.answer = False
            self.question_number += 1

    def fetch_image(self, event):
        self.im = Image.open(self.img_file_path)
        self.im.thumbnail(self.img_size)
        self.tk_im = ImageTk.PhotoImage(self.im)

    def display_image(self, event):
        self.canvas.create_image(350, 0, anchor="nw", image=self.tk_im, tag="question")

    def save_answer(self, event):
        self.previous_guesses.append(self.guess)

    def check_answer(self, event):
        self.previous_guesses.append(self.guess.get())
        return self.guess.get().lower() == self.pokemon.lower()

    def create_user(self, event):
        """
        Creates user in dict
        """
        self.user_dict[self.username.get()] = []
        self.is_userCreated = True

    def get_user_scores(self, username):
        """
        Returns list of previous scores on the user
        """
        return self.user_dict[self.username.get()]

    def save_user_score(self):
        """
        Saves user score to list of scores
        """
        if not self.is_userCreated:
            self.user_dict[self.username.get()] = []
        self.user_dict[self.username.get()].append(self.score)
        
    def save_scores(self):
        """
        Saves user dictionary with scores
        """
        pickle.dump(self.user_dict, open("../pickles/user_score_dictionary.p", "wb"))
"""
from tkthread import tk, TkThread

root = QuizGame()        # create the root window
tkt = TkThread(root)  # make the thread-safe callable

import threading, time
def run(func):
    threading.Thread(target=func).start()

run(lambda:     root.wm_title('FAILURE'))
run(lambda: tkt(root.wm_title,'SUCCESS'))

root.update()
time.sleep(5)  # _tkinter.c:WaitForMainloop fails
root.mainloop()
"""


app = QuizGame()
app.mainloop()
