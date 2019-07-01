# Who's That Pokemon

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


## **Conclusion**

## **What's Next?**

## **Photo and Data Credits**