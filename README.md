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
<p>Dataset imported from : https://www.artstor.org/</p>


<b>3. Data exploration and data cleaning: </b>
<div>The dataset contains information about Museums and Educational institutions around the world to gather around 800 images of paintings, 400 for both the Song and Ming dynasties.</div>

  ![ming_vs_song](https://user-images.githubusercontent.com/78753719/155208873-76d55264-528a-4899-b0dd-684b0d2a0799.png)
  
### General Approach	
Our project falls in the domain of supervised machine learning, given a ArtStor dataset of 800 famous Ming and Song dynasty artists and their paintings. We will firstly process our raw data, ie the full-sized images of paintings, by resizing and removing noise from the dataset. For example, some paintings are sketchwork and hence not very relevant for the purpose of this project.

We will trained a Convolutional Neural Network to recognize the different time periods.

Last but not the least, we will continue to improve our model and summarize what could be the limitations of our project.

<!-- * Ranking painters per number of paintings 
<div>On the 50 represented painters, the 3 painters with the most paintings in this dataset are Vincent Van Gogh, Edgar Degas and Pablo Picasso, all of them having more than 400 paintings. </div>

![](https://github.com/Camillelib/Art_Classifying_Project/blob/master/Media/3.%20Painters.png?raw=true)

* Nationalities of painters
<div>The dataset contains 17 different nationalities. French, Dutch and Spanish having the most paintings on this dataset. </div>

![](https://github.com/Camillelib/Art_Classifying_Project/blob/master/Media/2.%20Nationalities.png?raw=true)

* Genres of painters
<div>There are a lot of different genres in this dataset, some painters being associated to several genres.</div>
<div>E.g. Impressionism and Post-Impressionism</div>

![](https://github.com/Camillelib/Art_Classifying_Project/blob/master/Media/4.%20Genres%20before%20cleaning.png?raw=true)

<div>After cleaning the dataset and attributing unique genres to each painter, 16 genres represent all artists and paintings.</div>

![](https://github.com/Camillelib/Art_Classifying_Project/blob/master/Media/5.%20Genres%20after%20cleaning.png?raw=true)

<b> 4. Image transformation </b>
All the images were resized and converted to numpy arrays. The genres (impressionism, baroque) were also converted to numpy arrays.

![](https://github.com/Camillelib/Art_Classifying_Project/blob/master/Media/9.%20Resizing%20images.png?raw=true)

<b> 5. Model building </b>

* 1st model
A first model was built using only one random genre, Baroque, to test the keras library. The accuracy obtained was of 100% since the paintings could only be baroque.

![](https://github.com/Camillelib/Art_Classifying_Project/blob/master/Media/6.%20Model%201.png?raw=true)

* 2nd model
From the 1st model, a new model was built using all 16 genres with the same structure. However, the first results for this model were really low, with only 22% accuracy.

![](https://github.com/Camillelib/Art_Classifying_Project/blob/master/Media/7.%20Model%202.png?raw=true)

* Final model:
A few improvements were made for the final model:

  * Only the top 5 genres (per number of paintings) were selected, i.e. Impressionism, Renaissance, Post-Impressionism, Symbolism and Baroque
  
  ![](https://github.com/Camillelib/Art_Classifying_Project/blob/master/Media/8.%20Model%203.png?raw=true)
  
  * Some transformations were performed on training images and added to the set. They consist of 90 degree rotation, random horizontal and vertical flips, and random zoom on overall 1000+ images.
  
  ![](https://github.com/Camillelib/Art_Classifying_Project/blob/master/Media/9b.%20Transforming%20images.png?raw=true)
  
  * Finally, parameters from the keras.Sequential model were modified, including: adding layers, modifying parameters, and adding a validation split. 
  
![](https://github.com/Camillelib/Art_Classifying_Project/blob/master/Media/10.%20Model%203%20summary.png?raw=true) -->

## Results
<!-- The result of this last model were improved, with 57% accuracy rate. 

![](https://github.com/Camillelib/Art_Classifying_Project/blob/master/Media/11.%20Model%203%20results.png?raw=true)

![](https://github.com/Camillelib/Art_Classifying_Project/blob/master/Media/12.%20Model%203%20confusion%20matrix.png?raw=true)

<div>Test:</div>

[![Image](/Media/13.Youtube_video.png)](https://www.youtube.com/watch?v=RbJoAtRr6hY&feature=youtu.be) -->


## Conclusion
<!-- Learnings:
* How to work with images in data analysis
* How to build neural network for image classification

Future improvements:
* Add  images,  painters,  new genres
* Work on a new dataset where images are already classified per genre (and not painters)
* Test other machine learning models on this dataset -->

