# Who's That Pokemon?

![banner](https://github.com/NJacobsohn/Whos_That_Pokemon/blob/master/img/whos_that_banner.jpg)

## Table of Contents

1. [Overview](#overview)
2. [Questions](#questions)
3. [Data](#cleaning)
4. [Setup](#setup)
5. [Visualization](#visualization)
6. [The Website](#the-website)
7. [Conclusion](#conclusion)
8. [What's Next?](#what's-next?)
9. [Photo and Data Credits](#photo-and-data-credits)

## **Overview**

One fateful afternoon on April 1st, 1997, the Pokemon animated show first reared its head to the public after a very successful year for the video games. The universal love and hype for this franchise was picking up and it was time for the Pokemon company to capitalize on that. The Pokemon television show was and still is fun, light hearted, and family friendly. So what are some ways that the Pokemon company made kids *love* their franchise? Obviously with easily recognizable and distinct characters. What about making the main protaganist (Ash Ketchum) 11 years old so that kids watching could easily relate and see themselves as a true Pokemon master?  

While those are pretty obvious, my favorite retention method they used was a little segment each episode called "Who's That Pokemon?". This segment had 2 parts to it. Before a commercial break a silhouette of a Pokemon would be shown and a question asked: Who's that Pokemon? Then when the show returned from its commercial break, they would reveal the Pokemon. This made it so easy to want to keep watching the show and, even if the episode was sub-par, watch it until the end to get that smug satisfaction of guessing what the Pokemon was.

I'm happy to report that this project is about minimizing the crushing feeling of defeat when you don't actually know "Who's That Pokemon?" and leave you with the smug sense of satisfaction every commercial break.

## **Questions**

- How well Pokemon could be classified with a homemade convolutional neural network (CNN), and how that compared to a known, tested CNN like Xception?
- How well do these networks scale as the number of classifiers increase from the original 151 to all 809 currently released?  

## **Data**

For the original 151 Pokemon, a wonderful kaggle user named [HarshitDwivedi](https://www.kaggle.com/thedagger) uploaded a [decently sanitized collection](https://www.kaggle.com/thedagger/pokemon-generation-one) of images. Each Pokemon had anywhere from 75-150 images in various formats and resolutions. I wrote a script to change all of the randomly generated files names (16 character hexadecimal) to the name of the Pokemon and an index number, as well as changing all of the file types to .jpeg. All other image augmentation such as resizing was done through Keras' ImageDataGenerator class.

## **Setup**

Xception didn't get much tuning due to time constraints (thank you Amazon), so I ran it with imagenet weights with 200x200 resolution images on a GPU enhanced Amazon EC2 instance with Sagemaker. My personal CNN got a lot of tuning because I was able to run it in a reasonable amount of time on my personal machine. Each network trained for 10 epochs. I built my network in a class to allow for modular layer architecture, when I build the model, I can choose a number of blocks of layers to run it with. The network architecture looked like this:

    Block 0:
    Convolutional2D (16 filters)
    TanH Activation
    Convolutional2D (16 filters)
    TanH Activation
    MaxPooling2D (2x2 pooling size)
    Dropout (0.25)

    Block 1+:
    SeparableConvolutional2D (16 + (32 * blocknumber) filters)
    TanH Activation
    SeparableConvolutional2D (16 + (32 * blocknumber) filters)
    TanH Activation
    MaxPooling2D (2x2 pooling size)
    Dropout (0.25)

    Final Block:
    Dense (128 neurons)
    Relu Activation
    Dense (149 neurons, # of classes)
    Softmax Activation

    Compiled with:
    Loss: Categorical Crossentropy
    Optimizer: Adam
    Metrics: Accuracy

I tested 10 epochs with various architectures with my CNN and the best performance I found was with the above hyperparameters and just one repeatable block. So the final model had 4 convolutional layers with 16 filters on the first 2 and 48 filters on the rest. My model was trained on 64x64 resolution images, but otherwise the same exact train/validation/holdout split as Xception was.  

## **Visualization**

Since the main thing I wanted to compare was how the models did versus each other, I wanted to plot a (rather large) confusion matrix for each model as well as how the accuracy and loss changed.  

Lets look at which one of these bad boys is the smartest (higher accuracy increases per epoch).

![my_cnn_acc_per_epoch](https://github.com/NJacobsohn/Whos_That_Pokemon/blob/master/img/accuracy_per_epoch_my_cnn.png)

![xception_acc_per_epoch](https://github.com/NJacobsohn/Whos_That_Pokemon/blob/master/img/accuracy_per_epoch_xception.png)

While this is cool, it's much more interesting to see them side by side each other.

![accuracy_comparison](https://github.com/NJacobsohn/Whos_That_Pokemon/blob/master/img/accuracy_comparison.png)

Despite Xception seeing a 5% performance increase over my CNN, it didn't really learn anything special. Its best epoch was the very first one, and it only got worse from there. The testing loss was always higher than my CNN, which made me believe that my net actually learned the data better than Xception did.  

Lets take a look at the loss:  

![loss_comparison](https://github.com/NJacobsohn/Whos_That_Pokemon/blob/master/img/loss_comparison.png)

But maybe they're better at predicting different things? Let's look at their confusion matrices.  

![my_cnn_cm](https://github.com/NJacobsohn/Whos_That_Pokemon/blob/master/img/my_cnn_cm.png)

![xception_cm](https://github.com/NJacobsohn/Whos_That_Pokemon/blob/master/img/xception_cm.png)

Xception did only a slightly better job with the first generation Pokemon versus my CNN. While the diagonal is better (more correct guesses), it looks like it has some serious problems with certain Pokemon. So Xception is (out of the box) better at its job than my home built one was, but is my network potentially better at classifying Pokemon that Xception missed?

    My CNN:
    Machoke for Beedrill 36.36% of the time
    Eevee for Dodrio 33.33% of the time
    Primeape for Mankey 33.33% of the time
    Vaporeon for Seadra 33.33% of the time
    Poliwrath for Poliwhirl 30.76% of the time 

    Xception:
    Magmar for Charmeleon 30.77% of the time
    Pidgeotto for Lickitung 30.77% of the time
    Pinsir for Nidoqueen 30.77% of the time
    Vileplume for Slowpoke 26.67% of the time
    Venonat for Venomoth 25% of the time

Now some of these make sense, like Poliwrath and Poliwhirl:

Poliwrath                  |  Poliwhirl
:-------------------------:|:-------------------------:
![poliwrath](https://github.com/NJacobsohn/Whos_That_Pokemon/blob/master/img/poliwrath.png)  |  ![poliwhirl](https://github.com/NJacobsohn/Whos_That_Pokemon/blob/master/img/poliwhirl.png)

Or Primeape and Mankey:

Primeape                  |  Mankey
:-------------------------:|:-------------------------:
![primeape](https://github.com/NJacobsohn/Whos_That_Pokemon/blob/master/img/primeape.png)  |  ![mankey](https://github.com/NJacobsohn/Whos_That_Pokemon/blob/master/img/mankey.png)

But the top mixup was Machoke for Beedrill and that does NOT make much sense:

Machoke                    |  Beedrill
:-------------------------:|:-------------------------:
![machoke](https://github.com/NJacobsohn/Whos_That_Pokemon/blob/master/img/machoke.png)  |  ![beedrill](https://github.com/NJacobsohn/Whos_That_Pokemon/blob/master/img/beedrill.png)

Xception seems to be all over the place, with two sensible pairs of Magmar and Charmeleon and Venonat and Venomoth:

Magmar                   |  Charmeleon
:-------------------------:|:-------------------------:
![magmar](https://github.com/NJacobsohn/Whos_That_Pokemon/blob/master/img/magmar.png)  |  ![charmeleon](https://github.com/NJacobsohn/Whos_That_Pokemon/blob/master/img/charmeleon.png)

Venonat                  |  Venomoth
:-------------------------:|:-------------------------:
![venonat](https://github.com/NJacobsohn/Whos_That_Pokemon/blob/master/img/venonat.png)  |  ![venomoth](https://github.com/NJacobsohn/Whos_That_Pokemon/blob/master/img/venomoth.png)

But the rest of the pairs don't make much human sense (I'm sure Xception would have a great explanation for these if it could talk):

Vileplume                  |  Slowpoke
:-------------------------:|:-------------------------:
![vileplume](https://github.com/NJacobsohn/Whos_That_Pokemon/blob/master/img/vileplume.png)  |  ![slowpoke](https://github.com/NJacobsohn/Whos_That_Pokemon/blob/master/img/slowpoke.png)

Pinsir                  |  Nidoqueen
:-------------------------:|:-------------------------:
![pinsir](https://github.com/NJacobsohn/Whos_That_Pokemon/blob/master/img/pinsir.png)  |  ![nidoqueen](https://github.com/NJacobsohn/Whos_That_Pokemon/blob/master/img/nidoqueen.png)

## **The Website**

I build a flask app and put it into a Docker container to allow for easy deployment. The application allows you to upload an image (ideally of a 1st generation Pokemon) and my model will classify it for you. Below is a screenshot of the homepage and what a prediction might look like.

![homepage](https://github.com/NJacobsohn/Whos_That_Pokemon/blob/master/img/homepage.png)

![prediction](https://github.com/NJacobsohn/Whos_That_Pokemon/blob/master/img/prediction.png)

## **Conclusion**

Overall the networks seem to have (mostly) reasonable mixups with Pokemon within each evolution line getting mixed up with each other. I honestly expected Xception to be much better than my net considering how extensive it is, but it really fell flat. I even think with proper tuning, my network could outperform Xception for this specific task as Xception doesn't get any better the more it trains. Using the relatively small quantity of images per class that I'm using, these nets I believe would be much better suited classifying evolution lines instead of specific Pokemon.

Although, if you've never seen a single Pokemon in your life, a random guess only has a 0.6% chance of being correct, so my neural network that's correct ~25% of the time is over 40 times better at guessing than you would be! Xception is 50 times more accurate!  

As far as scaling in concerned, I feel like my accuracy would float around the same ~25% even with more Pokemon. I think it's more dependent on the amount of evolution lines added than the individual Pokemon.  

## **What's Next?**

- Run nets on one Pokemon from each evolution line rather than every single Pokemon.
- Gather images for the remaining ~650 Pokemon and re-run predictions on both nets.
- Re-train both nets with new, larger data sets and compare new accuracy with the new weights.
- No longer sulk in self pity because of incorrect guesses.

## **Photo and Data Credits**

- [Banner:](https://www.sporcle.com/games/Chenchilla/silhouettes) Thanks to this quiz site for having this as a quiz thumbnail.
- [Data:](https://www.kaggle.com/thedagger/pokemon-generation-one) Thanks to Kaggle and HarshitDwivedi for uploading images for the original 151.
- [All Pokemon Images:](https://archives.bulbagarden.net/wiki/) Thanks to bulbagarden's archive for having tons of pictures of all Pokemon.
- [Xception:](https://arxiv.org/abs/1610.02357) If you'd like to read more about Xception, the paper can be found here.
