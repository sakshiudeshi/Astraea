import random

import spacy
nlp = spacy.load('en')

# Add neural coref to SpaCy's pipe
import neuralcoref
neuralcoref.add_to_pipe(nlp)
pass

def predict_clusters(sentence):
    doc = nlp(sentence)
    if doc._.has_coref: 
        return (doc._.coref_resolved, doc._.coref_clusters)
    else:
        return ('', '')

## Example of an error
print(predict_clusters('The carpenter asked the assistant if he can play a ukelele'))
print(predict_clusters('The carpenter asked the assistant if she can play a ukelele'))