# Classification and Sentiment Analysis model 
***
This project involves the implementation of transformer models made available through the fastai and Hugging Face transformers libraries. We wanted to train a model that will take in a coherent passage and return the subject of the passage and its average connotation (a number bounded between -1 and 1).  We achieved this through text categorization datasets and sentiment classification. This collaborative project also served as an introduction to a wide variety of ML and NLP topics.

Contributors: Josh Dawson, Ezra Crowe, Ben Toker

## Previous archive
This is a continuation of an archived project which can be found
[here](https://github.com/jedawson04/WinterTerm2024-NLPModel).

## Installation and Setup
***
Python is the main language used in this project. To install the required dependencies use pip as shown below.
```
$ cd nlpmodel
$ pip install -r requirements.txt
```

To use the trained models, you will need to download them from [Google Drive](https://drive.google.com/drive/folders/1CwxtrDH2olbzOznYaLuzfGEmLP26scT_?usp=sharing), as GitHub cannot store these large files here. 

## Classification Setup
Download the classification model (found under ``Models/Classification_Model``) and place it inside the repository folder. 
Now run the *classifyinput.py* script. Now you can demo any input you want! 

## Sentiment setup
Download the sentiment model (found under ``Models/Sentiment_Model_2.2``) and place it inside the repository folder.
You should be able to enter a novel passage in the *sentiment.ipynb* notebook and get a reasonably accurate prediction of sentiment back.
***

### Issues and further direction
The goal of this project was ultimately to learn about natural language processing and to give ourselves an introduction to machine learning. What you see
here is the minimum viable product that we sought to create. Further direction that we discussed included having a streamlined front-end for the classification and sentiment models, though this ultimately was not prioritized. Both models achieve around 86-90% accuracy from our testing, but could be improved in theory. We also wanted to scale up the data that the classification model is trained on, as it only uses one niche categorization dataset (reuters21578) so you can only enter inputs that would more or less match its odd labels. Using topic modeling on generic datasets or simply including more categorization data would imply an improvement in this area. 
