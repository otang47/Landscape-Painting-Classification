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


<img width="1512" alt="Screen Shot 2022-05-05 at 4 18 50 PM" src="https://user-images.githubusercontent.com/78753719/167019198-b442d5ca-e790-4e41-9639-491da2d031d7.png">


## Conclusion
<!-- Learnings:
* How to work with images in data analysis
* How to build neural network for image classification

Future improvements:
* Add  images,  painters,  new genres
* Work on a new dataset where images are already classified per genre (and not painters)
* Test other machine learning models on this dataset -->

