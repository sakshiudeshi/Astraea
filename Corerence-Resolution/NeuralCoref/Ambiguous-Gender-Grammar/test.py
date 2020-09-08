import random

import spacy
nlp = spacy.load('en')

# Add neural coref to SpaCy's pipe
import neuralcoref
neuralcoref.add_to_pipe(nlp)
pass

doc = nlp(u'My sister has a dog. She loves him.')
