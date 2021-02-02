
# Astraea
Code for ["Astraea: Grammar-based Fairness Testing"](https://arxiv.org/abs/2010.02542). In this repository, we present code for fairness of three
NLP tasks, [Coreference Resolution](https://demo.allennlp.org/coreference-resolution), 
[Sentiment Analysis](https://demo.allennlp.org/sentiment-analysis) and 
[Masked Language Modeling](https://demo.allennlp.org/masked-lm?text=The%20doctor%20ran%20to%20the%20emergency%20room%20to%20see%20%5BMASK%5D%20patient.)

###  Coreference Resolution
The code for the fairness testing of Coreference Resolution (coref) can be found in the `Coreference-Resolution` folder. We test three coref NLP algorithms. Deep-learning based [Neuralcoref](https://github.com/huggingface/neuralcoref), [AllenNLP](https://demo.allennlp.org/coreference-resolution/coreference-resolution) and rule-based [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/)

Please see the respective pages for detailed installation instructions. 

For each coreference resolution module, we have two grammar variants. They are the ambiguous and unambiguous grammars. The ambiguous variant tests for fairness related to occupation, gender and religion whereas the unambiguous variant tests only for fairness with respect to gender. Additionally, for easy reproducilbility and verification we provide all the generated pickles and tokens. These analysis script can be run in the Data-Analysis files for each grammar variant. 

### Sentiment Analysis

We evaluate 11 (6 pre-trained, 5 self-trained) sentiment analysis models. The self-trained models can be found in the folder `Sentiment Analysis/trained-sentiment-analyzers`. Please refer to the [paper](https://arxiv.org/abs/2010.02542) for details of the self-trained models. 
Astraea elavulates the following pre-trained sentiment analysis models:
* [Pattern Analysis TextBlob](https://textblob.readthedocs.io/en/dev/)
* [NaiveBayes TextBlob](https://textblob.readthedocs.io/en/dev/)
* [NLTK-Vader](https://www.nltk.org/_modules/nltk/sentiment/vader.html)
* [Vader Sentiment](https://pypi.org/project/vaderSentiment/)
* [Google NLP](https://cloud.google.com/natural-language)
* [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/)

Please refer to the specific page for installation instructions. 


### Masked Language Modelling
We evaluated [bert-cased](https://huggingface.co/bert-base-cased), [bert-uncased](https://huggingface.co/bert-base-uncased), [distilbert-cased](https://huggingface.co/distilbert-base-cased) and [distilbert-uncased](https://huggingface.co/distilbert-base-uncased). Please refer to the [Huggingface](https://huggingface.co/) page for further documentation. These models must be stored in the `Masked-Language-Modelling/models` folder

As with the other cases, we provide the tokens and the errors in pickle files for easy reproduction. These are stored in the folder `Masked-Language-Modelling/saved_pickles`

This repository is still under development. Please email sakshi_udeshi@mymail.sutd.edu.sg for any questions.
