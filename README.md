# Chinese Landscape Painting Classification Using a Convolutional Neural Networks


This project was done by Olivia Tang and Rose William for the University of Pennsylvania's CIS 107: Visual Culture through the Computer's Eye. 


## Abtract
In this project, we investigated image classification as a high-level computer vision task with implications for art history. We utilized roughly 600 photos from publicly available databases to explore if convolutional neural networks could be trained to classify pre-modern artworks into their respective eras of origin. Following are our results of using a multi-layered neural network on a large dataset of digitized images of Song and Ming Dynasty landscape paintings. 

## Introduction
Historically, the fine arts and their study has been confined to expert eyes who are charged with discerning their origins, era, artist, and influences.  Now, with the introduction and democratization of large digitized fine art collections into the public domain through the efforts of museums and universities, art can now be scrutinized not just through connoisseurship but as big data. Inspired by the success of Convolutional Neural Networks (CNN) in image classification projects, we used CNN to categorize our assembled large fine art datasets on Song and Ming Dynasty Chinese landscape paintings into their respective chronological origin. Our reasoning behind doing so was driven by our curiosity at what the neural network would use to discern the two eras, or if it could even discern stylistic differences perhaps imperceptible to untrained human eyes. In order to examine the capabilities of the deep model in fine-art painting classification, we trained an end-to-end deep convolutional model pulling our dataset from the publicly accessible large-scale “ArtStore” dataset comprising over 2.5 million paintings and eventually building a network with an accuracy of approximately 70%. 

## Process 

### Librairies used
* torch
* torchvision
* glob
* PIL
* os

### Steps
<b>1. Data collection</b>

<div>50% of our dataset was sourced from ArtStor, a fine arts and humanities database that has over 2.5 million pieces of artwork from museums, archives, and educational institutions all around the world. This dataset is considered a curational cornerstone and is constantly updated by reputable organizations, making it a suitable source of images for our experiment. Pulling from the ArtStor database, we were able to create a dataset of 211 Ming Dynasty and 229 Song Dynasty landscape paintings for a total of 440 images as well as the accompanying metadata relating to the paintings’ artist, style, genre, and origin. However, after creating a CNN with this dataset, we had reason to believe that it was too small to create conclusive results. We then supplemented this dataset with more images pulled from web image search scraping in order to create a more robust dataset. Our final dataset resulted in 322 Song Dynasty paintings and 311 Ming Dynasty paintings for a total of 633 images. Approximately 80% of those images were used in the training set (506) and 20% were used in the testing set (127). To fine-tune our dataset, we narrowed our ArtStore searches to include only ink and watercolor artworks that depicted primarily landscape scenes on paper. By doing so, we deliberately filtered out other styles of Song and Ming Dynasty art like furniture and calligraphy in order to eliminate as many confounding variables as possible and thus allowing for the direct comparison of Song and Ming Dynasty landscape paintings. We defined Song Dynasty art as originating from the years 960 until 1279 and Ming Dynasty art from 1368 until 1644. Below are examples of Song and Ming Dynasty landscape paintings pulled from our database.</div>

  ![ming_vs_song](https://user-images.githubusercontent.com/78753719/155208873-76d55264-528a-4899-b0dd-684b0d2a0799.png)
  
### General Approach	
Our network itself is based on PyTorch’s CIFAR 10 classifier. We transformed our dataset images to the dimensions of 32x32 and defined batch size as 4 to fit the structure of our CNN which has two convolutional layers, one max-pooling layer, and three fully connected layers. A rough flowchart of this can be seen in Figure 1. After transforming the images into normalized tensors, we split the dataset into training and testing images. To train the network, we looped over our data iterator to feed our training images into the network to optimize for 50 epochs. We then tested the resulting network on our testing dataset and received a 69% accuracy.

![CNN arch](https://user-images.githubusercontent.com/78753719/167018475-9f416a44-27f4-400c-a716-ea484f5ae89d.png)



## Results
The results yielded a 68% accuracy for the testing set, with 78.6% for Ming Dynasty paintings and 60% for Song Dynasty Paintings. It also yielded 88% accuracy on the training set. Given that the original dataset was roughly equal in both Song Dynasty and Ming Dynasty inputs (50.8% Song Dynasty paintings and 49.8% Ming Dynasty Paintings), this is significantly better than random chance. The difficulty of this computer-vision task and the lack of reputable data sources proved to be hurdles to accuracy. While the results are not high enough to compete with human experts, there is reasonable evidence to believe that there are discernable differences between Song and Ming Dynasty landscape paintings that can be identified by neural networks. 

<img width="637" alt="Screen Shot 2022-05-05 at 4 23 54 PM" src="https://user-images.githubusercontent.com/78753719/167019508-fe8db504-87e6-479a-9cfa-52abc1bdad4f.png">

<img width="624" alt="Screen Shot 2022-05-05 at 2 33 54 PM" src="https://user-images.githubusercontent.com/78753719/167019242-cfe89aef-a464-4a64-9dc9-68592d5c7c5b.png">



## Conclusion
Fine-art categorization could supplement or possibly replace human labor in categorizing or authenticating artworks as digital artworks.  Machine learning has the ability to use a bigger quantity of data to categorize a broader spectrum of art much faster than a person which is especially relevant with the growth of online art auctions and art ownership that resides entirely in the digital domain like non-fungible tokens. Additionally, the automation of art classification could help art museums build out online exhibitions in which paintings are ordered in chronological progression allowing the viewer to see the continuity and change of style and form across eras. That being said, there is still great value in human interpretation that should not be thrown away especially as art historians seek to understand the cultural and social implications of stylistic changes throughout the era. More research is needed to fully harness the potential of computer vision and use it to benefit the current work of art historians and museum curators. 

## Next Steps
The next steps for the experiment would be to look at different, more advanced neural networks that are more suited for art classification purposes. Perhaps this is one of the reasons why our algorithm did not reach a high degree of accuracy to the very rudimentary nature of the network we were utilizing. It could also be beneficial to go over images that were misclassified and try and see if there is any obvious pattern to these misclassified images. The experiment could also transition to look at Chinese paintings that have a larger chronological gap between them which could potentially make it easier for the algorithm to classify them since the stylistic differences would be more significant.

