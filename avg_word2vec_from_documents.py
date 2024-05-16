import gensim
from gensim import utils
import numpy as np
import sys
from sklearn.datasets import fetch_20newsgroups
from nltk import word_tokenize
from nltk import download
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
get_ipython().magic('matplotlib inline')

#model Google News, run once to download pre-trained vectors
#!wget https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
download('punkt') #tokenizer, run once
download('stopwords') #stopwords dictionary, run once
stop_words = stopwords.words('english')

def preprocess(text):
    text = text.lower()
    doc = word_tokenize(text)
    doc = [word for word in doc if word not in stop_words]
    doc = [word for word in doc if word.isalpha()] #restricts string to alphabetic characters only
    return doc

#doc content  -> num label     -> string label
#note to self: texts[XXXX] -> y[XXXX] = ZZZ -> ng20.target_names[ZZZ]
# Fetch ng20 dataset
ng20 = fetch_20newsgroups(subset='all',
                          remove=('headers', 'footers', 'quotes'))
#text and ground truth labels
texts, y = ng20.data, ng20.target
corpus = [preprocess(text) for text in texts]


# Remove empty docs
def filter_docs(corpus, texts, labels, condition_on_doc):
    """
    Filter corpus, texts and labels given the function condition_on_doc which takes
    a doc.
    The document doc is kept if condition_on_doc(doc) is true.
    """
    number_of_docs = len(corpus)

    if texts is not None:
        texts = [text for (text, doc) in zip(texts, corpus)
                 if condition_on_doc(doc)]

    labels = [i for (i, doc) in zip(labels, corpus) if condition_on_doc(doc)]
    corpus = [doc for doc in corpus if condition_on_doc(doc)]

    print("{} docs removed".format(number_of_docs - len(corpus)))

    return (corpus, texts, labels)
    
corpus, texts, y = filter_docs(corpus, texts, y, lambda doc: (len(doc) != 0))

# Remove OOV words and documents with no words in model dictionary
def document_vector(word2vec_model, doc):
    # remove out-of-vocabulary words
    doc = [word for word in doc if word in word2vec_model.vocab]
    return np.mean(word2vec_model[doc], axis=0)

def has_vector_representation(word2vec_model, doc):
    """check if at least one word of the document is in the
    word2vec dictionary"""
    return not all(word not in word2vec_model.vocab for word in doc)
    
corpus, texts, y = filter_docs(corpus, texts, y, lambda doc: has_vector_representation(model, doc))

x =[]
for doc in corpus: #look up each doc in model
    x.append(document_vector(model, doc))
    
X = np.array(x) #list to array

np.save('documents_vectors.npy', X)  #np.savetxt('documents_vectors.txt', X)
np.save('labels.npy', y)             #np.savetxt('labels.txt', y)

# Plot 2 PCA components
pca = PCA(n_components=2)
x_pca = pca.fit_transform(X)

plt.figure(1, figsize=(30, 20),)
plt.scatter(x_pca[:, 0], x_pca[:, 1],s=100, c=y, alpha=0.2)
X_tsne = TSNE(n_components=2, verbose=2).fit_transform(X)
plt.figure(1, figsize=(30, 20),)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1],s=100, c=y, alpha=0.2)
