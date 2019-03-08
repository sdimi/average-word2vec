### Average words to represent documents with word2vec
Quick Python script I wrote in order to process the 20 Newsgroup dataset with word embeddings. Suggested to run on a Jupyter Notebook. Most word2vec [1] pre-trained models allow to get numerical representations of individual words but not of entire documents. While most sophisticated methods like doc2vec exist, with this script we simply average each word of a document so that the generated document vector is actually a centroid of all words in feature space.

![title](https://github.com/sdimi/average-word2vec/blob/master/workflow.png)

### Dependencies
``gensim`` (for word2vec model load)

``numpy`` (for averaging and array manipulation)
#### Optional
``nltk`` (for text pre-processing)

``sklearn`` (for dataset load)

``matplotlib`` (for plotting)


### Why
For the representation of text as numbers, there are many options out there. The simplest methodology when dealing with text is to create a word frequency matrix that simply counts the occurrence of each word. A variant of this method is to estimate the log scaled frequency of each word, but considering its occurrence in all documents (tf-idf). Also another popular option is to take into account the context around each word (n-grams), so that e.g. New York is evaluated as a bi-gram and not separately. However, these methods do not capture high level semantics of text, just frequencies. A recent advance on the field of Natural Language Processing proposed the use of word embeddings. Word embeddings are dense representations of text, coming through a feed-forward neural network. That way, each word is being represented by a point that is embedded in the high-dimensional space. With careful training, words that can be used interchangeably should have similar embeddings. A popular word embeddings network is word2vec. Word2vec is a simple, one-hidden-layer neural network that sums word embeddings and instead of minimizing a multi-class logistic loss (softmax), it minimizes a binary logistic loss on positive and negative samples, allowing to handle huge vocabularies efficiently.

In order to represent the 20Newsgroup documents, I use a pre-trained word2vec model provided by Google. This model was trained on 100 billion words of Google News and contains 300-dimensional vectors for 3 million words and phrases. As a pre-processing, the 20Newsgroups dataset was tokenized and the English stop-words were removed. Empty documents were removed (555 documents deleted). Documents with not at least 1 word in word2vec model were removed (9 documents deleted). The final resulting dataset consists of 18282 documents. For each document, the mean of the embeddings of each word was calculated, so that each document is represented by a 300-dimensional vector.

The newsgroup dataset was retrieved via its [helper function](http://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html) from the Python library scikit-learn . The pre-trained word2vec model is available [here](https://code.google.com/archive/p/word2vec/). In order to process the model, the gensim library was used.

This code was developed as part of the data pre-processing section for my paper on interactive dimensionality reduction [2] and I thought it might be useful for other people dealing with text representation tasks. 

> [1]  Mikolov, Tomas, Ilya Sutskever, Kai Chen, Greg S. Corrado, and Jeff Dean. ["Distributed representations of words and  phrases and their compositionality."](https://scholar.google.gr/citations?user=oBu8kMMAAAAJ&hl=en&oi=sra#d=gs_md_cita-d&u=%2Fcitations%3Fview_op%3Dview_citation%26hl%3Den%26user%3DoBu8kMMAAAAJ%26citation_for_view%3DoBu8kMMAAAAJ%3ACB2v5VPnA5kC%26tzom%3D0) In Advances in neural information processing systems, pp. 3111-3119. 2013.

> [2] Spathis, Dimitris, Nikolaos Passalis, and Anastasios Tefas. ["Interactive dimensionality reduction using similarity projections."](https://www.sciencedirect.com/science/article/pii/S0950705118305677) Knowledge-Based Systems 165 (2019): 77-91.

