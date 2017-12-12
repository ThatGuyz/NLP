# NLP
Programmation neuro-linguistique

### easy way : https://explosion.ai/blog/chatbot-node-js-spacy




# Resources

* Natural Language Toolkit (NLTK): The complete toolkit for all NLP techniques.
* Pattern – A web mining module for the with tools for NLP and machine learning.
* TextBlob – Easy to use nl p tools API, built on top of NLTK and Pattern.
* spaCy – Industrial strength N LP with Python and Cython.
* Gensim – Topic Modelling for Humans
* Stanford Core NLP – NLP services and packages by Stanford NLP Grou
* Scikit-learn: Machine learning in Python
#
# Some more useful resources on chatbots:
* http://www.wildml.com/2016/04/deep-learning-for-chatbots-part-1-introduction/
* https://venturebeat.com/2016/08/01/how-deep-reinforcement-learning-can-help-chatbots/
* http://web.stanford.edu/class/cs124/lec/chatbot.pdf

#
* More resources on Tensorflow:
* http://lauragelston.ghost.io/speakeasy-pt2/
* https://speakerdeck.com/inureyes/building-ai-chat-bot-using-python-3-and-tensorflow


## Cycle

* Downlaod DataSet
* Create a model
* Train this shit 
* test out it

#https://github.com/llSourcell/tensorflow_chatbot

# Example 1 
* python -m spacy download fr

>import spacy
nlp_fr = spacy.load('fr')
doc_fr = nlp_fr(u'Hello, world. Here are two sentences.')

#Example 1 
>import spacy
>import random

>nlp = spacy.load('en')
>train_data = [("Uber blew through $1 million", {'entities': [(0, 4, 'ORG')]})]

>with nlp.disable_pipes(*[pipe for pipe in nlp.pipe_names if pipe != 'ner']):
 >   optimizer = nlp.begin_training()
  >  for i in range(10):
   >     random.shuffle(train_data)
    >    for text, annotations in train_data:
     >       nlp.update([text], [annotations] sgd=optimizer)
> nlp.to_disk('/model')



#Example NLTK
# # Self-Learning from sentence
from nltk import word_tokenize, pos_tag
text = "I am learning Natural Language Processing on Analytics Vidhya"
tokens = word_tokenize(text)
print pos_tag(tokens)
">>> [('I', 'PRP'), ('am', 'VBP'), ('learning', 'VBG'), ('Natural', 'NNP'),('Language', 'NNP'),
('Processing', 'NNP'), ('on', 'IN'), ('Analytics', 'NNP'),('Vidhya', 'NNP')]"
