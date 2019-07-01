# Who's That Pokemon

![banner](https://github.com/NJacobsohn/Whos_That_Pokemon/blob/master/img/whos_that_banner.jpg)

# Table of Contents
1. [Overview](#overview)
2. [Questions](#questions)
3. [Data](#cleaning)
4. [Visualization](#visualization)
5. [Conclusion](#conclusion)
6. [What's Next?](#what's-next?)
7. [Photo and Data Credits](#photo-and-data-credits)



## **Overview**
One fateful afternoon on April 1st, 1997, the Pokemon animated show first reared it's head to the public after a very successful year for the video games. The universal love and hype for this franchise was picking up and it was time for the Pokemon company to capitalize on that. The Pokemon television show was and still is fun, light hearted, and family friendly. So what are some ways that the Pokemon company made kids *love* their franchise? Obviously with easily recognizable and distinct characters. What about making the main protaganist (Ash Ketchum) 11 years old so that kids watching could easily relate and see themselves as a true Pokemon master? 

While those are pretty obvious, my favorite retention method they used was a little segment each episode called "Who's That Pokemon?". This segment had 2 parts to it. Before a commercial break a silouhette of a Pokemon would be shown and a question asked: Who's that Pokemon? Then when the show returned from its commercial break, they would reveal the Pokemon. This made it so easy to want to keep watching the show and, even if the episode was sub-par, watch it until the end to get that smug satisfaction of guessing what the Pokemon was.

Well I'm happy to report that this project is about minimizing the crushing feeling of defeat when you don't actually know "Who's That Pokemon?" and leave you with the smug sense of satisfaction every commercial break.

## **Questions**
- How well Pokemon could be classified with a homemade convolutional neural network (CNN), and how that compared to a known, tested CNN like Xception?
- How well do these networks scale as the number of classifiers increase? (From the original 151 to all 809 currently released) 

## **Data**
For the original 151 Pokemon, a wonderful kaggle user named [HarshitDwivedi](https://www.kaggle.com/thedagger) uploaded a [decently sanitized collection](https://www.kaggle.com/thedagger/pokemon-generation-one) of images. Each Pokemon had anywhere from 75-150 images in various formats and resolutions. I wrote a script to change all of the randomly generated files names (16 character hexadecimal) to the name of the Pokemon and an index number, as well as changing all of the file types to .jpeg. All other image augmentation such as resizing was done through keras' ImageDataGenerator class.

## **Visualization**
Since the main thing I wanted to compare was how the models did versus each other, I wanted to plot a (rather large) confusion matrix for each model as well as how the accuracy and loss changed. 

So lets look at which one of these bad boys is the smartest (higher accuracy increases per epoch)

![my_cnn_acc_per_epoch](linktomy_acc_per_epoch)


![xception_acc_per_epoch](linktoxception_acc_per_epoch)

Description of which model learned quicker goes here. Dummy text YEAAAAAAAH

Let's take a look at the confusion matrices.

![my_cnn_cm](https://github.com/NJacobsohn/Whos_That_Pokemon/blob/master/img/my_cnn_cm.png)


![xception_cm](linktoxception_cm)

Xception did a fantastic job with the first generation Pokemon versus my CNN.
So Xception is clearly better at its job than my home built one was, but is my network potentially better at classifying Pokemon that Xception missed?

![data_about_common_mixups_xception](link_to_mixup_data_xception)

    
    Machoke for Beedrill 36.36% of the time
    Eevee for Dodrio 33.33% of the time
    Primeape for Mankey 33.33% of the time
    Vaporeon for Seadra 33.33% of the time
    Poliwrath for Poliwhirl 30.76% of the time
    
Now some of these make sense, like Poliwrath and Poliwhirl.


Poliwrath                  |  Poliwhirl
:-------------------------:|:-------------------------:
![](https://github.com/NJacobsohn/Whos_That_Pokemon/blob/master/img/poliwrath.png)  |  ![](https://github.com/NJacobsohn/Whos_That_Pokemon/blob/master/img/poliwhirl.png)


Or Primeape and Mankey.


Primeape                  |  Mankey
:-------------------------:|:-------------------------:
![](https://github.com/NJacobsohn/Whos_That_Pokemon/blob/master/img/primeape.png)  |  ![](https://github.com/NJacobsohn/Whos_That_Pokemon/blob/master/img/mankey.png)

But the top mixup was Machoke for Beedrill and that does NOT make much sense to me.

Machoke                    |  Beedrill
:-------------------------:|:-------------------------:
![](https://github.com/NJacobsohn/Whos_That_Pokemon/blob/master/img/machoke.png)  |  ![](https://github.com/NJacobsohn/Whos_That_Pokemon/blob/master/img/beedrill.png)

## **Conclusion**

## **What's Next?**
- Gather images for the remaining ~650 Pokemon and re-run predictions on both nets
- Re-train both nets with new, larger data sets and compare new accuracy with the new weights
- No longer sulk in self pity because you guessed incorrectly.

## **Photo and Data Credits**

- [Banner:](https://www.sporcle.com/games/Chenchilla/silhouettes) Thanks to this quiz site for having this as a quiz thumbnail.
- [All Pokemon Images:](https://archives.bulbagarden.net/wiki/) Thanks to bulbagarden's archive for having tons of pictures of all Pokemon