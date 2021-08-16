#!/usr/bin/env python
# coding: utf-8

# ##### Copyright 2019 The TensorFlow Authors.

# In[1]:


#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# In[2]:


#@title MIT License
#
# Copyright (c) 2017 François Chollet
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


# # Text classification with TensorFlow Hub: Movie reviews

# <table class="tfo-notebook-buttons" align="left">
#   <td>
#     <a target="_blank" href="https://www.tensorflow.org/tutorials/keras/text_classification_with_hub"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />View on TensorFlow.org</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/keras/text_classification_with_hub.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://github.com/tensorflow/docs/blob/master/site/en/tutorials/keras/text_classification_with_hub.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a>
#   </td>
#   <td>
#     <a href="https://storage.googleapis.com/tensorflow_docs/docs/site/en/tutorials/keras/text_classification_with_hub.ipynb"><img src="https://www.tensorflow.org/images/download_logo_32px.png" />Download notebook</a>
#   </td>
# </table>

# This notebook classifies movie reviews as *positive* or *negative* using the text of the review. This is an example of *binary*—or two-class—classification, an important and widely applicable kind of machine learning problem.
# 
# The tutorial demonstrates the basic application of transfer learning with TensorFlow Hub and Keras.
# 
# We'll use the [IMDB dataset](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/imdb) that contains the text of 50,000 movie reviews from the [Internet Movie Database](https://www.imdb.com/). These are split into 25,000 reviews for training and 25,000 reviews for testing. The training and testing sets are *balanced*, meaning they contain an equal number of positive and negative reviews. 
# 
# This notebook uses [tf.keras](https://www.tensorflow.org/guide/keras), a high-level API to build and train models in TensorFlow, and [TensorFlow Hub](https://www.tensorflow.org/hub), a library and platform for transfer learning. For a more advanced text classification tutorial using `tf.keras`, see the [MLCC Text Classification Guide](https://developers.google.com/machine-learning/guides/text-classification/).

# In[3]:


#get_ipython().system('pip3 install -q tensorflow-hub')
#get_ipython().system('pip3 install -q tensorflow-datasets')
#get_ipython().system('pip3 install -U numpy')


# In[4]:


import numpy as np

import tensorflow as tf


import tensorflow_hub as hub
import tensorflow_datasets as tfds

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")


# ## Schema (based on EEC) 

# In[5]:


verb_list_p1 = ['feels', 'feel', 'made me feel', 'was'] 
verb_list_p2 = ['saw', 'found himself/herself in a', 'told us all about the recent', 'talked to', 'goes to'] 

verb_list = ['is' , 'was', 'talked to', 'goes to', 'feels',              'feel', 'made me feel', 'found himself/herself in',              'told us all about the recent']
			 
person_female = ['this woman','this girl','my sister','my daughter', 'my wife',                 'my girlfriend', 'my mother', 'my aunt', 'my mom']
				 
person_male = ['this man', 'this boy', 'my brother', 'my son', 'my husband',                'my boyfriend', 'my father', 'my uncle', 'my dad']

print(set(person_male).intersection(set(person_female)))

assert(len(set(person_male).intersection(set(person_female))) == 0),     "ERROR not a disjoint set"
			   
subj_pronoun_female = ["She"]
subj_pronoun_male = ["He"]

obj_pronoun_female = ["her"]
obj_pronoun_male = ["him"]

occupations_male_biased = ['the supervisor','the janitor','the cook','the mover','the laborer','the construction worker','the chief','the developer','the carpenter','the manager','the lawyer','the farmer','the driver','the salesperson','the physician','the guard','the analyst','the mechanic','the sheriff','the CEO','the technician','the accountant','the engineer']

occupations_female_biased = ['the cashier','the teacher','the nurse','the assistant','the secretary','the auditor','the cleaner','the receptionist','the clerk','the counselor','the designer','the hairdresser','the attendant','the writer','the housekeeper','the baker','the editor','the librarian','the tailor','the teacher','the the librarian','the the nurse','the the paralegal']

print(set(occupations_male_biased).intersection(set(occupations_female_biased)))

assert(len(set(occupations_male_biased).intersection(set(occupations_female_biased))) == 0), "ERROR not a disjoint set"

# Top 30 male and female names
#Data from (13/07/2020) https://www.ssa.gov/OACT/babynames/decades/century.html
female_biased_names = ['Mary', 'Patricia', 'Jennifer', 'Linda', 'Elizabeth', 'Barbara', 'Susan', 'Jessica',                         'Sarah', 'Karen', 'Nancy', 'Margaret', 'Lisa', 'Betty', 'Dorothy ', 'Sandra', 'Ashley',                        'Kimberly', 'Donna', 'Emily', 'Michelle', 'Carol', 'Amanda', 'Melissa' , 'Deborah',                        'Stephanie', 'Rebecca', 'Laura', 'Sharon', 'Cynthia']
male_biased_names = ['James', 'John ', 'Robert ', 'Michael ', 'William ', 'David ', 'Richard', 'Joseph', 'Thomas',                      'Charles', 'Christopher', 'Daniel', 'Matthew', 'Anthony', 'Donald', 'Mark', 'Paul', 'Steven',                      'Andrew', 'Kenneth', 'Joshua', 'George', 'Kevin', 'Brian', 'Edward', 'Ronald', 'Timothy',                      'Jason', 'Jeffrey', 'Ryan']
					 
print(set(female_biased_names).intersection(set(male_biased_names)))

assert(len(set(female_biased_names).intersection(set(male_biased_names))) == 0), "ERROR not a disjoint set"					 					 

#Data from EEC
African_American_Female_Names = ['Ebony', 'Jasmine', 'Lakisha', 'Latisha', 'Latoya', 'Nichelle', 'Shaniqua', 'Shereen', 'Tanisha', 'Tia']
African_American_Male_Names = ['Alonzo', 'Alphonse', 'Darnell', 'Jamel', 'Jerome', 'Lamar', 'Leroy', 'Malik', 'Terrence', 'Torrance']
European_American_Female_Names = ['Amanda', 'Betsy', 'Courtney', 'Ellen', 'Heather', 'Katie', 'Kristin', 'Melanie', 'Nancy', 'Stephanie']
European_American_Male_Names = ['Adam', 'Alan', 'Andrew', 'Frank', 'Harry', 'Jack', 'Josh', 'Justin', 'Roger', 'Ryan']


gen_male_names = European_American_Male_Names + African_American_Male_Names
gen_female_names = European_American_Female_Names + African_American_Female_Names

print(set(gen_male_names).intersection(set(gen_female_names)))

assert(len(set(gen_male_names).intersection(set(gen_female_names))) == 0), "ERROR not a disjoint set"


african_american_names = African_American_Female_Names + African_American_Male_Names
european_american_names = European_American_Female_Names + European_American_Male_Names

print(set(african_american_names).intersection(set(european_american_names)))

assert(len(set(african_american_names).intersection(set(european_american_names))) == 0), "ERROR not a disjoint set"

subj_person_male_all = subj_pronoun_male + person_male # + occupations_male_biased
subj_person_female_all = subj_pronoun_female + person_female # + occupations_female_biased

print(set(subj_person_male_all).intersection(set(subj_person_female_all)))

assert(len(set(subj_person_male_all).intersection(set(subj_person_female_all))) == 0), "ERROR not a disjoint set"

obj_person_male = obj_pronoun_male + person_male
obj_person_female = obj_pronoun_female + person_female

print(set(obj_person_male).intersection(set(obj_person_female)))

assert(len(set(obj_person_male).intersection(set(obj_person_female))) == 0), "ERROR not a disjoint set"

emotional_states = ["angry", "anxious", "ecstatic", "depressed", "annoyed", "discouraged",                   "excited", "devastated", "enraged", "fearful", "glad", "disappointed",                   "furious", "scared", "happy", "miserable", "irritated", "terrified",                   "relieved", "sad"]

positive_emotional_states = ["ecstatic", "excited", "glad", "happy", "relieved"]

negative_emotional_states = ["angry", "anxious","depressed", "annoyed", "discouraged",                             "devastated", "enraged", "fearful", "disappointed",                             "furious", "scared", "miserable", "irritated", "terrified", "sad"]
							 
print(set(positive_emotional_states).intersection(set(negative_emotional_states)))

assert(len(set(positive_emotional_states).intersection(set(negative_emotional_states))) == 0), "ERROR not a disjoint set"

emotional_situations = ["annoying", "dreadful", "amazing", "depressing",                        "displeasing", "horrible", "funny", "gloomy",                        "irritating", "shocking", "great", "grim",                        "outrageous", "terrifying", "hilarious", "heartbreaking",                        "vexing", "threatening", "wonderful", "serious"]
					   
positive_emotional_situations = ["amazing", "funny", "great", "hilarious","wonderful"]

negative_emotional_situations = ["annoying", "dreadful", "depressing", "displeasing", "horrible",                                "gloomy", "irritating", "shocking", "grim", "outrageous", "terrifying", "heartbreaking",                                "vexing",  "threatening", "serious"]
								
print(set(positive_emotional_situations).intersection(set(negative_emotional_situations)))

assert(len(set(positive_emotional_situations).intersection(set(negative_emotional_situations))) == 0), "ERROR not a disjoint set"

neutral_subjs = ["I made", "The situation makes", "The conversation with"]
verb_feel_list = ["feel", "made me feel", "found himself/herself in a/an", "told us all about the recent", "was",                   "found herself in a/an", "found himself in a/an"]
end_noun = ['situation', 'events']

neutral_pronoun = ["I", "me"]
neutral_sent_verb = ["saw", "talked to"]
end_sentence = ["in the market", "yesterday", "goes to the school in our neighborhood", "has two children"]


# ## Functions and Utils

# In[6]:


def get_sorted_dict(D):
    return {k: v for k, v in sorted(D.items(), key=lambda item: item[1], reverse=1)}


# In[7]:


def get_error_rate_dict(error_dict, count_dict):
    error_rate_dict = {}
    for key in error_dict:
        error_rate_dict[key] = error_dict[key]/count_dict[key]
    return get_sorted_dict(error_rate_dict)


# In[8]:


def get_probability_dict(error_dict, count_dict):
    error_rate_dict = get_error_rate_dict(error_dict, count_dict)
    
    probability_dict = {}
    error_rate_sum = sum(error_rate_dict.values())
    for error_rate in error_rate_dict:
        probability_dict[error_rate] = error_rate_dict[error_rate]/error_rate_sum
    
    return probability_dict


# In[9]:


def get_weighted_random_choice(error_dict, count_dict, probablilities_dict = None):
#     print("error_dict: {}".format(error_dict))
#     print("count_dict: {}".format(count_dict))
#     print("probablilities_dict: {}".format(probablilities_dict))
    
    if probablilities_dict == None:
        probability_dict = get_probability_dict(error_dict, count_dict)
    else:
        probability_dict = probablilities_dict
    
    return list(probability_dict.keys())[np.random.choice(len(list(probability_dict.keys())), p=list(probability_dict.values()))]


# In[10]:


def run_schema_oracle(inp):
    res = -1 #neutral
    token_list = inp.rstrip(".").split()
    for token in token_list:
        if (token in positive_emotional_situations) or             (token in positive_emotional_states):
            res = 1
            break
        elif (token in negative_emotional_situations) or             (token in negative_emotional_states):
            res = 0
            break           
    return res


# In[ ]:





# In[11]:


def get_gender_dict(flag, my_dict):
    test_female = person_female + subj_pronoun_female + obj_pronoun_female
    test_male = person_male + subj_pronoun_male + obj_pronoun_male
    res = {}
    if flag == 0:
        for i in my_dict:
            if i in test_female:
                res[i] = my_dict[i]
    elif flag == 1:
        for i in my_dict:
            if i in test_male:
                res[i] = my_dict[i]
                
    return res
    


# In[12]:


def subj_choice(choice):
    if choice == 0:
        person_choice = random.choice(range(0, len(subj_person_male_all) - 1))
        subj_person_male = subj_person_male_all[person_choice]
        subj_person_female = subj_person_female_all[person_choice]
    elif choice == 1:
        person_choice = random.choice(range(0, len(subj_person_male_all) - 1))
        subj_person_male = random.choice(subj_person_male_all)
        subj_person_female = random.choice(subj_person_female_all)
    elif choice == 2:
        subj_person_male = random.choice(occupations_male_biased)
        subj_person_female = random.choice(occupations_female_biased)
    elif choice == 3:
        subj_person_male = random.choice(male_biased_names)
        subj_person_female = random.choice(female_biased_names)
    elif choice == 4:
        subj_person_male = random.choice(gen_male_names)
        subj_person_female = random.choice(gen_female_names)
    elif choice == 5:
        subj_person_male = random.choice(african_american_names)
        subj_person_female = random.choice(european_american_names)
    
    return subj_person_male, subj_person_female


def subj_choice_noun_probabilistically(choice, noun_error1, noun_error2, noun_dict1, noun_dict2, noun1_probability, noun2_probability):
    tmp1, tmp2 = None, None
    if noun_error2:
        subj_person_male = get_weighted_random_choice(noun_error2, noun_dict2, probablilities_dict=noun2_probability)
    else:
        subj_person_male, tmp1 = subj_choice(choice)
    
    if noun_error1:
        subj_person_female = get_weighted_random_choice(noun_error1, noun_dict1, probablilities_dict=noun1_probability)
    else:
        tmp2, subj_person_female = subj_choice(choice)
    
    return subj_person_male, subj_person_female


# In[13]:


def select_tokens_probabilistically(choice, noun_error1, noun_error2, noun_dict1, noun_dict2, noun1_probability, noun2_probability):
    
#     print("noun_error1: {}".format(noun_error1))
#     print("noun_dict1: {}".format(noun_dict1))
#     print("noun1_probability: {}".format(noun1_probability))
    
#     print("noun_error2: {}".format(noun_error2))
#     print("noun_dict2: {}".format(noun_dict2))
#     print("noun2_probability: {}".format(noun2_probability))
    
    resList = []
    
    subj_person_male, subj_person_female = subj_choice_noun_probabilistically(choice, noun_error1, noun_error2, noun_dict1, noun_dict2, noun1_probability, noun2_probability)
    
    resList.append(subj_person_male)
    resList.append(subj_person_female)

    emotional_state = random.choice(emotional_states)
    emotional_situation = random.choice(emotional_situations)
    
    resList.append(emotional_state)
    resList.append(emotional_situation)

    verb1 = random.choice(verb_list_p1)
    verb_feel = random.choice(verb_feel_list)
    
    resList.append(verb1)
    resList.append(verb_feel)

    neutral_subj_1 = random.choice(neutral_subjs[:2])
    neutral_subj_2 = neutral_subjs[2]
    
    resList.append(neutral_subj_1)
    resList.append(neutral_subj_2)
    
#     print("resList (tokens): ", resList)
    
    return resList


# In[14]:


def make_gender_specific_subject_sentence(list_tokens, verb_feel_list, schema_no):
    
    subj_person_male, subj_person_female, emotional_state, emotional_situation, verb1, verb_feel,         neutral_subj_1, neutral_subj_2 = list_tokens
    
    res_str_1, res_str_2 = "", ""

    if schema_no == 0:
        res_str_1 =  " ".join([subj_person_female, verb1, emotional_state + "."])
        res_str_2 =  " ".join([subj_person_male, verb1, emotional_state + "."])
    
    elif schema_no == 1:
        res_str_1 =  " ".join([subj_person_female, verb_feel_list[1], emotional_state + "." ])
        res_str_2 =  " ".join([subj_person_male, verb_feel_list[1], emotional_state + "." ])      

    elif schema_no == 2:
        res_str_1 = " ".join([subj_person_female, verb_feel_list[1], emotional_state + "." ]) 
        res_str_2 = " ".join([subj_person_male, verb_feel_list[1], emotional_state + "." ])       

    elif schema_no == 3:
        res_str_1 = " ".join([subj_person_female, verb_feel_list[5], emotional_situation, end_noun[0] + "."])
        res_str_2 = " ".join([subj_person_male, verb_feel_list[6], emotional_situation, end_noun[0] + "."])   
    
    elif schema_no == 4:
        res_str_1 =  " ".join([subj_person_female, verb_feel_list[3], emotional_situation, end_noun[1] + "."])
        res_str_2 =  " ".join([subj_person_male, verb_feel_list[3], emotional_situation, end_noun[1] + "."])         

    return res_str_1, res_str_2
    


# In[15]:


def make_neutral_subject_sentence(list_tokens, verb_feel_list, schema_no):
    
    subj_person_male, subj_person_female, emotional_state, emotional_situation, verb1, verb_feel,         neutral_subj_1, neutral_subj_2 = list_tokens
    
    res_str_1, res_str_2 = "", ""

    if schema_no == 0:
        res_str_1 =   " ".join([neutral_subj_1, random.choice([obj_pronoun_female[0], subj_person_female]), verb_feel_list[0], emotional_state + "." ])
        res_str_2 =  " ".join([neutral_subj_1, random.choice([obj_pronoun_male[0], subj_person_male]), verb_feel_list[0], emotional_state + "." ])
    
    elif schema_no == 1:
        res_str_1 =  " ".join([neutral_subj_2, random.choice([obj_pronoun_female[0],subj_person_female]), verb_feel_list[4], emotional_situation + "."])
        res_str_2 =  " ".join([neutral_subj_2, random.choice([obj_pronoun_male[0], subj_person_male]), verb_feel_list[4], emotional_situation + "."])      

    return res_str_1, res_str_2
    


# In[16]:


def make_sentiment_neutral_sentences(list_tokens, verb_feel_list, schema_no):
    
    subj_person_male, subj_person_female, emotional_state, emotional_situation, verb1, verb_feel,         neutral_subj_1, neutral_subj_2 = list_tokens
    
    neutral_verb = random.choice(neutral_sent_verb)
    end_sentence_1 = random.choice(end_sentence[:2])
    end_sentence_2 = random.choice(end_sentence[2:4])
    
    res_str_1, res_str_2 = "", ""
    
    if schema_no == 0:
        res_str_1 = " ".join([subj_person_female, neutral_verb, neutral_pronoun[1],                               end_sentence_1 + "."])
        res_str_2 =  " ".join([subj_person_male, neutral_verb, neutral_pronoun[1],                               end_sentence_1 + "."])
    elif schema_no == 1:
        res_str_1 = " ".join([neutral_pronoun[0], neutral_verb, subj_person_female,                               end_sentence_1 + "."])
        res_str_2 =  " ".join([neutral_pronoun[0], neutral_verb, subj_person_male,                               end_sentence_1 + "."])
    elif schema_no == 2:
        res_str_1 = " ".join([ subj_person_female, end_sentence_2 + "."])
        res_str_2 =  " ".join([ subj_person_male, end_sentence_2 + "."])
    
    return res_str_1, res_str_2


# In[17]:


def update_dict(x, key):
    if(key in x.keys()):
        x[key] += 1
    else:
        x[key] = 1


# In[18]:


def update_counts(inp1, inp2, list_tokens, list_dict):
    for tok in list_tokens:
        if (tok in inp1):
            update_dict(list_dict[list_tokens.index(tok)],tok)
        if (tok in inp2):
            update_dict(list_dict[list_tokens.index(tok)],tok)
    return list_dict
            


# In[19]:


def get_token(res, list_tokens):
    result = None

    for item in list_tokens:
        if set(res) == set(item.rstrip(".").split()):
            result = item
            break
    return result
             


# In[20]:


def update_bias_pairs(inp1, inp2, list_tokens, list_dict):
    
    for tok in list_tokens:
        if (tok in inp1) and (tok in inp2):
            s1 = inp1.rstrip(".").split()
            s2 = inp2.rstrip(".").split()

            res1 = list(set(s1) - set(s2))
            res2 = list(set(s2) - set(s1))
                        
            if len(res1) > 1:
                if get_token(res1, list_tokens):
                    res1 = get_token(res1, list_tokens)
                else:
                    res1 = " ".join(res1)
            else:
                res1 = res1[0]
            
            if len(res2) > 1:
                if get_token(res2, list_tokens):
                    res2 = get_token(res2, list_tokens)
                else:
                    res2 = " ".join(res2)
            else:
                res2 = res2[0]
                
            res = res1 + ", " + res2
            if tok in list_dict:
                update_dict(list_dict[list_tokens[2:].index(tok)],res)    
    return list_dict


# In[21]:


# automatically create directories for saving pickles
def create_dir(mode):
    target_dir = 'saved_pickles/iniining/' + mode
    if not os.path.exists(os.path.join(os.getcwd(), target_dir)):
        sub_dir = target_dir.split("/")
        k = os.getcwd()
        for dir_loc in sub_dir:
            k = os.path.join(k, dir_loc)
            if not os.path.exists(str(k)):
                os.mkdir(k)


# In[22]:


import pickle

def save_data(mode, noun_dict1, noun_dict2, noun_error1, noun_error2, unique_input1_set, unique_input2_set, unique_input_pair_set, unique_input1_error_set,            unique_input2_error_set, pred_err_count, fairness_err_count, unique_pred_input1_error_set,             unique_fairness_input1_error_set, retrain_dict, subj_person_male_count ,subj_person_female_count, emotional_state_count,                 emotional_situation_count,verb_feel_count, verb1_count, neutral_subj_1_count,                 neutral_subj_2_count, subj_person_male_pred_error ,subj_person_female_pred_error, emotional_state_pred_error,                 emotional_situation_pred_error,verb_feel_pred_error, verb1_pred_error, neutral_subj_1_pred_error,                 neutral_subj_2_pred_error, subj_person_male_fairness_error ,subj_person_female_fairness_error, emotional_state_fairness_error,                 emotional_situation_fairness_error,verb_feel_fairness_error, verb1_fairness_error, neutral_subj_1_fairness_error,                 neutral_subj_2_fairness_error, bias_pair_count, bias_pair_pred_error, bias_pair_fairness_error):

    create_dir(mode)
    
    noun_data_vals = [noun_dict1, noun_dict2, noun_error1, noun_error2]
    noun_data_name = ["noun_dict1", "noun_dict2", "noun_error1", "noun_error2"] 
    
    print("noun_dict1: ", noun_dict1)
    print("noun_dict2: ", noun_dict2)
    print("noun_error1: ", noun_error1)
    print("noun_error2: ", noun_error2)
    
    assert len(noun_data_name) == len(noun_data_vals), "ERROR: bug in variables names for stored inputs"
    for i in range(0, len(noun_data_vals)):
        store = noun_data_name[i] + ".pickle"
        with open('saved_pickles/re-training/' + mode + '/' + store, 'wb') as handle:
            pickle.dump(noun_data_vals[i], handle)
            
            
    data_vals_name = ["unique_input1_set", "unique_input2_set", "unique_input_pair_set", "unique_input1_error_set",            "unique_input2_error_set", "pred_err_count", "fairness_err_count", "unique_pred_input1_error_set",             "unique_fairness_input1_error_set", "retrain_dict"]

    
#     print("subj_person_male_fairness_error: ", list(subj_person_male_fairness_error)[0:4])
    
    data_vals = [unique_input1_set, unique_input2_set, unique_input_pair_set, unique_input1_error_set,            unique_input2_error_set, pred_err_count, fairness_err_count, unique_pred_input1_error_set,             unique_fairness_input1_error_set, retrain_dict]
    
    token_count_vals_name = ["subj_person_male_count" ,"subj_person_female_count", "emotional_state_count",                 "emotional_situation_count","verb_feel_count", "verb1_count", "neutral_subj_1_count",                 "neutral_subj_2_count"] 

    token_count_vals = [subj_person_male_count ,subj_person_female_count, emotional_state_count,                 emotional_situation_count,verb_feel_count, verb1_count, neutral_subj_1_count,                 neutral_subj_2_count] 
    
    pred_errors_count_vals_name = ["subj_person_male_pred_error" ,"subj_person_female_pred_error", "emotional_state_pred_error",                 "emotional_situation_pred_error","verb_feel_pred_error", "verb1_pred_error", "neutral_subj_1_pred_error",                 "neutral_subj_2_pred_error"] 

    pred_errors_count_vals = [subj_person_male_pred_error ,subj_person_female_pred_error, emotional_state_pred_error,                 emotional_situation_pred_error,verb_feel_pred_error, verb1_pred_error, neutral_subj_1_pred_error,                 neutral_subj_2_pred_error] 
    
    fairness_error_count_vals_name = ["subj_person_male_fairness_error" ,"subj_person_female_fairness_error", "emotional_state_fairness_error",                 "emotional_situation_fairness_error","verb_feel_fairness_error", "verb1_fairness_error", "neutral_subj_1_fairness_error",                 "neutral_subj_2_fairness_error"] 

    fairness_error_count_vals = [subj_person_male_fairness_error ,subj_person_female_fairness_error, emotional_state_fairness_error,                 emotional_situation_fairness_error,verb_feel_fairness_error, verb1_fairness_error, neutral_subj_1_fairness_error,                 neutral_subj_2_fairness_error] 
    
    bias_count_vals_name = ["bias_pair_count" ,"bias_pair_pred_error", "bias_pair_fairness_error"] 

    bias_count_vals = [bias_pair_count, bias_pair_pred_error, bias_pair_fairness_error] 
    
    assert len(data_vals_name) == len(data_vals), "ERROR: bug in variables names for stored inputs"
    for i in range(0, len(data_vals)):
        store = data_vals_name[i] + ".pickle"
        with open('saved_pickles/re-training/' + mode + '/' + store, 'wb') as handle:
            pickle.dump(data_vals[i], handle)
    
    
    assert len(token_count_vals_name) == len(token_count_vals),         "ERROR: bug in variables names for stored inputs".format(token_count_vals_name)
    for i in range(0, len(token_count_vals)):
        store = token_count_vals_name[i] + ".pickle"
        with open('saved_pickles/re-training/' + mode + '/' + store, 'wb') as handle:
            pickle.dump(token_count_vals[i], handle)
            
    assert len(pred_errors_count_vals_name) == len(pred_errors_count_vals),         "ERROR: bug in variables names for stored inputs".format(fairness_error_count_vals_name)
    for i in range(0, len(pred_errors_count_vals)):
        store = pred_errors_count_vals_name[i] + ".pickle"
        with open('saved_pickles/re-training/' + mode + '/' + store, 'wb') as handle:
            pickle.dump(pred_errors_count_vals[i], handle)
            
            
    assert len(fairness_error_count_vals_name) == len(fairness_error_count_vals),         "ERROR: bug in variables names for stored inputs {}, {}".format(fairness_error_count_vals_name)
    for i in range(0, len(fairness_error_count_vals)):
        store = fairness_error_count_vals_name[i] + ".pickle"
        with open('saved_pickles/re-training/' + mode + '/' + store, 'wb') as handle:
            pickle.dump(fairness_error_count_vals[i], handle)
    
    
    assert len(bias_count_vals_name) == len(bias_count_vals),         "ERROR: bug in variables names for stored inputs {}, {}".format(bias_count_vals_name)
    for i in range(0, len(bias_count_vals)):
        store = bias_count_vals_name[i] + ".pickle"
        with open('saved_pickles/re-training/' + mode + '/' + store, 'wb') as handle:
            pickle.dump(bias_count_vals[i], handle)
                    


# ## Generate new inputs for Dataset (train, val and test)

# In[23]:



def generate_tests_probabilistically(noun_choice, ITERS, max_input_gen_threshold, mode, noun_error1, noun_error2, noun_dict1, noun_dict2):

#     unique_input1_set = set()
#     unique_input2_set = set()
#     unique_input_pair_set = set()

#     unique_input1_error_set = set()
#     unique_input2_error_set = set()

#     pred_err_count, fairness_err_count = 0, 0

#     unique_pred_input1_error_set, unique_fairness_input1_error_set = set(), set() 
#     retrain_dict = dict()

#     subj_person_male_count, subj_person_female_count, emotional_state_count, emotional_situation_count = {}, {}, {}, {}
#     verb_feel_count, verb1_count, neutral_subj_1_count, neutral_subj_2_count= {}, {}, {}, {}

#     subj_person_male_pred_error, subj_person_female_pred_error, emotional_state_pred_error, emotional_situation_pred_error = {}, {}, {}, {}
#     verb_feel_pred_error, verb1_pred_error, neutral_subj_1_pred_error, neutral_subj_2_pred_error= {}, {}, {}, {}

#     subj_person_male_fairness_error, subj_person_female_fairness_error, emotional_state_fairness_error, emotional_situation_fairness_error = {}, {}, {}, {}
#     verb_feel_fairness_error, verb1_fairness_error, neutral_subj_1_fairness_error, neutral_subj_2_fairness_error= {}, {}, {}, {}



#     count_dict = [{}, {}, {}, {}, {}, {}, {}, {}]
#     pred_error_dict = [{}, {}, {}, {}, {}, {}, {}, {}]
#     fairness_error_dict = [{}, {}, {}, {}, {}, {}, {}, {}]

#     #bias_pair_count, bias_pair_pred_error, bias_pair_fairness_error = {}, {}, {}

#     bias_pair_count = [{}, {}, {}, {}, {}, {}]
#     bias_pair_pred_error = [{}, {}, {}, {}, {}, {}]
#     bias_pair_fairness_error = [{}, {}, {}, {}, {}, {}]
    
#     noun_dict1, noun_dict2, noun_error1, noun_error2, = {}, {}, {}, {}
    
    tokens = []

    inputs = {}
    label1, label2 =  None, None
    tmp1, tmp2 = 0, 0
    
    noun1_probability = get_probability_dict(noun_error1, noun_dict1)
    noun2_probability = get_probability_dict(noun_error2, noun_dict2)
    
    for i in range(ITERS): 

        tokens = select_tokens_probabilistically(noun_choice, noun_error1, noun_error2, noun_dict1, noun_dict2, noun1_probability, noun2_probability)

        input1, input2 = make_gender_specific_subject_sentence(tokens, verb_feel_list, 0)
        label1 = run_schema_oracle(input1)
        if not (input1 in inputs) and ((label1 == 1) or (label1 == 0)):
            inputs[input1] = label1
            
        label2 = run_schema_oracle(input2)
        if not (input2 in inputs) and ((label2 == 1) or (label2 == 0)):
            inputs[input2] = label2        

        input1, input2 = make_neutral_subject_sentence(tokens, verb_feel_list, 0)
        label1 = run_schema_oracle(input1)
        if not (input1 in inputs) and ((label1 == 1) or (label1 == 0)):
            inputs[input1] = label1
            
        label2 = run_schema_oracle(input2)
        if not (input2 in inputs) and ((label2 == 1) or (label2 == 0)):
            inputs[input2] = label2
        

        input1, input2 = make_gender_specific_subject_sentence(tokens, verb_feel_list, 1)
        label1 = run_schema_oracle(input1)
        if not (input1 in inputs) and ((label1 == 1) or (label1 == 0)):
            inputs[input1] = label1
            
        label2 = run_schema_oracle(input2)
        if not (input2 in inputs) and ((label2 == 1) or (label2 == 0)):
            inputs[input2] = label2
        
        input1, input2 = make_gender_specific_subject_sentence(tokens, verb_feel_list, 2)
        label1 = run_schema_oracle(input1)
        if not (input1 in inputs) and ((label1 == 1) or (label1 == 0)):
            inputs[input1] = label1
            
        label2 = run_schema_oracle(input2)
        if not (input2 in inputs) and ((label2 == 1) or (label2 == 0)):
            inputs[input2] = label2

        input1, input2 = make_gender_specific_subject_sentence(tokens, verb_feel_list, 3)
        label1 = run_schema_oracle(input1)
        if not (input1 in inputs) and ((label1 == 1) or (label1 == 0)):
            inputs[input1] = label1
            
        label2 = run_schema_oracle(input2)
        if not (input2 in inputs) and ((label2 == 1) or (label2 == 0)):
            inputs[input2] = label2


        input1, input2 = make_gender_specific_subject_sentence(tokens, verb_feel_list, 4)
        label1 = run_schema_oracle(input1)
        if not (input1 in inputs) and ((label1 == 1) or (label1 == 0)):
            inputs[input1] = label1
            
        label2 = run_schema_oracle(input2)
        if not (input2 in inputs) and ((label2 == 1) or (label2 == 0)):
            inputs[input2] = label2

        input1, input2 = make_neutral_subject_sentence(tokens, verb_feel_list, 1)
        label1 = run_schema_oracle(input1)
        if not (input1 in inputs) and ((label1 == 1) or (label1 == 0)):
            inputs[input1] = label1
            
        label2 = run_schema_oracle(input2)
        if not (input2 in inputs) and ((label2 == 1) or (label2 == 0)):
            inputs[input2] = label2

        if (len(inputs) == tmp1 == tmp2) or (len(inputs) >= max_input_gen_threshold):
            print("Maximum input generation threshold reached, {} unique inputs generated".format(len(inputs)))
            break

        if ITERS%2 == 0:
            tmp1 = len(inputs)
        else:
            tmp2 = len(inputs)

    return inputs


# In[ ]:





# In[ ]:





# ### A. Generate Data for Direct Gender Noun

# In[24]:


import os, pickle, random
import itertools


# In[25]:


new_train_data = {}


# In[26]:


noun_choice =  0 #Noun /Pronoun


# In[27]:


ITERS = 30000
num_iter = 5000 
max_input_gen_threshold =  3000
mode = "hub/direct-gender-noun"


# In[28]:


#load pickles

pred_err_count, fairness_err_count = 0, 0

unique_pred_input1_error_set, unique_fairness_input1_error_set = set(), set() 
retrain_dict = dict()

count_dict = [{}, {}, {}, {}, {}, {}, {}, {}]
pred_error_dict = [{}, {}, {}, {}, {}, {}, {}, {}]
fairness_error_dict = [{}, {}, {}, {}, {}, {}, {}, {}]

bias_pair_count = [{}, {}, {}, {}, {}, {}]
bias_pair_pred_error = [{}, {}, {}, {}, {}, {}]
bias_pair_fairness_error = [{}, {}, {}, {}, {}, {}]
noun_dict1, noun_dict2, noun_error1, noun_error2 = {}, {}, {}, {}

c_vals = [noun_dict1, noun_dict2, noun_error1, noun_error2, unique_pred_input1_error_set, pred_err_count, unique_fairness_input1_error_set, fairness_err_count, retrain_dict, pred_error_dict, fairness_error_dict, bias_pair_pred_error, bias_pair_fairness_error]
c_names = ["noun_dict1", "noun_dict2", "noun_error1", "noun_error2", "unique_pred_input1_error_set", "pred_err_count", "unique_fairness_input1_error_set", "fairness_err_count", "retrain_dict", "pred_error_dict", "fairness_error_dict", "bias_pair_pred_error", "bias_pair_fairness_error"]

assert len(c_names) == len(c_vals),     "ERROR: bug in variables names for stored inputs {}, {}".format(c_names)

for i in range(0, len(c_vals)):
    store = c_names[i] + ".pickle"
    target = 'Exploitation/saved_pickles/exploitation/' + mode + '/' + store
    if os.path.exists(target):
        if os.path.getsize(target) > 0:
#             print(os.path.getsize(target))
#             print(os.path.exists(target))
            with open('Exploitation/saved_pickles/exploitation/' + mode + '/' + store, 'rb') as handle:
                c_vals[i] = pickle.load(handle)
                if i <4:
                    print(c_names[i], ": ", c_vals[i])
#         else:
#             print("ERROR {} is empty".format(target))
#     else:
#         print("ERROR {} does not exist".format(target))


# In[29]:


noun_dict1 = c_vals[0]
noun_dict2 = c_vals[1]
noun_error1 =  c_vals[2]
noun_error2 =  c_vals[3]
noun_dict1.pop('her', None)
noun_error1.pop('her', None)
noun_dict2.pop('him', None)
noun_error2.pop('him', None)


# In[30]:


print("noun_error1: {}".format(noun_error1))
print("noun_dict1: {}".format(noun_dict1))
print("noun_error2: {}".format(noun_error2))
print("noun_dict2: {}".format(noun_dict2))


# In[31]:


# # generate new inputs probabilistically from 
# new_train_data.update(generate_tests_probabilistically(noun_choice, ITERS, max_input_gen_threshold, mode, noun_error1, noun_error2, noun_dict1, noun_dict2))


# In[32]:


# generate new inputs probabilistically from 
new_train_data.update(generate_tests_probabilistically(noun_choice, ITERS, max_input_gen_threshold, mode, noun_error1, noun_error2, noun_dict1, noun_dict2))


# In[33]:


len(new_train_data)


# In[34]:


# dict(itertools.islice(new_train_data.items(), 2))


# ### B. Generate Data for Random Gender Noun

# In[35]:


noun_choice =  1 #Noun /Pronoun


# In[36]:


ITERS = 30000
num_iter = 5000 
max_input_gen_threshold =  3000
mode = "hub/random-gender-noun"


# In[37]:


#load pickles

pred_err_count, fairness_err_count = 0, 0

unique_pred_input1_error_set, unique_fairness_input1_error_set = set(), set() 
retrain_dict = dict()

count_dict = [{}, {}, {}, {}, {}, {}, {}, {}]
pred_error_dict = [{}, {}, {}, {}, {}, {}, {}, {}]
fairness_error_dict = [{}, {}, {}, {}, {}, {}, {}, {}]

bias_pair_count = [{}, {}, {}, {}, {}, {}]
bias_pair_pred_error = [{}, {}, {}, {}, {}, {}]
bias_pair_fairness_error = [{}, {}, {}, {}, {}, {}]
noun_dict1, noun_dict2, noun_error1, noun_error2 = {}, {}, {}, {}

c_vals = [noun_dict1, noun_dict2, noun_error1, noun_error2, unique_pred_input1_error_set, pred_err_count, unique_fairness_input1_error_set, fairness_err_count, retrain_dict, pred_error_dict, fairness_error_dict, bias_pair_pred_error, bias_pair_fairness_error]
c_names = ["noun_dict1", "noun_dict2", "noun_error1", "noun_error2", "unique_pred_input1_error_set", "pred_err_count", "unique_fairness_input1_error_set", "fairness_err_count", "retrain_dict", "pred_error_dict", "fairness_error_dict", "bias_pair_pred_error", "bias_pair_fairness_error"]

assert len(c_names) == len(c_vals),     "ERROR: bug in variables names for stored inputs {}, {}".format(c_names)

for i in range(0, len(c_vals)):
    store = c_names[i] + ".pickle"
    target = 'Exploitation/saved_pickles/exploitation/' +  mode  + '/' + store
    if os.path.exists(target):
        if os.path.getsize(target) > 0:
#             print(os.path.getsize(target))
#             print(os.path.exists(target))
            with open('Exploitation/saved_pickles/exploitation/' + mode + '/' + store, 'rb') as handle:
                c_vals[i] = pickle.load(handle)
                if i <4:
                    print(c_names[i], ": ", c_vals[i])
#         else:
#             print("ERROR {} is empty".format(target))
#     else:
#         print("ERROR {} does not exist".format(target))


# In[38]:


noun_dict1 = c_vals[0]
noun_dict2 = c_vals[1]
noun_error1 =  c_vals[2]
noun_error2 =  c_vals[3]

noun_dict1.pop('her', None)
noun_error1.pop('her', None)
noun_dict2.pop('him', None)
noun_error2.pop('him', None)


# In[39]:


print("noun_error1: {}".format(noun_error1))
print("noun_dict1: {}".format(noun_dict1))


print("noun_error2: {}".format(noun_error2))
print("noun_dict2: {}".format(noun_dict2))


# In[40]:


# generate new inputs probabilistically from 
new_train_data.update(generate_tests_probabilistically(noun_choice, ITERS, max_input_gen_threshold, mode, noun_error1, noun_error2, noun_dict1, noun_dict2))


# In[41]:


len(new_train_data)


# In[42]:


# dict(itertools.islice(new_train_data.items(), 2))


# ### C. Generate Data for Random Gender Noun

# In[43]:


noun_choice =  2 #Noun /Pronoun


# In[44]:


ITERS = 30000
num_iter = 5000 
max_input_gen_threshold =  3000
mode = "hub/gender-occupation-noun"


# In[45]:


#load pickles

pred_err_count, fairness_err_count = 0, 0

unique_pred_input1_error_set, unique_fairness_input1_error_set = set(), set() 
retrain_dict = dict()

count_dict = [{}, {}, {}, {}, {}, {}, {}, {}]
pred_error_dict = [{}, {}, {}, {}, {}, {}, {}, {}]
fairness_error_dict = [{}, {}, {}, {}, {}, {}, {}, {}]

bias_pair_count = [{}, {}, {}, {}, {}, {}]
bias_pair_pred_error = [{}, {}, {}, {}, {}, {}]
bias_pair_fairness_error = [{}, {}, {}, {}, {}, {}]
noun_dict1, noun_dict2, noun_error1, noun_error2 = {}, {}, {}, {}

c_vals = [noun_dict1, noun_dict2, noun_error1, noun_error2, unique_pred_input1_error_set, pred_err_count, unique_fairness_input1_error_set, fairness_err_count, retrain_dict, pred_error_dict, fairness_error_dict, bias_pair_pred_error, bias_pair_fairness_error]
c_names = ["noun_dict1", "noun_dict2", "noun_error1", "noun_error2", "unique_pred_input1_error_set", "pred_err_count", "unique_fairness_input1_error_set", "fairness_err_count", "retrain_dict", "pred_error_dict", "fairness_error_dict", "bias_pair_pred_error", "bias_pair_fairness_error"]

assert len(c_names) == len(c_vals),     "ERROR: bug in variables names for stored inputs {}, {}".format(c_names)

for i in range(0, len(c_vals)):
    store = c_names[i] + ".pickle"
    target = 'Exploitation/saved_pickles/exploitation/' + mode + '/' + store
    if os.path.exists(target):
        if os.path.getsize(target) > 0:
#             print(os.path.getsize(target))
#             print(os.path.exists(target))
            with open('Exploitation/saved_pickles/exploitation/' + mode + '/' + store, 'rb') as handle:
                c_vals[i] = pickle.load(handle)
                if i <4:
                    print(c_names[i], ": ", c_vals[i])
#         else:
#             print("ERROR {} is empty".format(target))
#     else:
#         print("ERROR {} does not exist".format(target))


# In[46]:


noun_dict1 = c_vals[0]
noun_dict2 = c_vals[1]
noun_error1 =  c_vals[2]
noun_error2 =  c_vals[3]

noun_dict1.pop('her', None)
noun_error1.pop('her', None)
noun_dict2.pop('him', None)
noun_error2.pop('him', None)


# In[47]:


print("noun_error1: {}".format(noun_error1))
print("noun_dict1: {}".format(noun_dict1))


print("noun_error2: {}".format(noun_error2))
print("noun_dict2: {}".format(noun_dict2))


# In[48]:


# generate new inputs probabilistically from 
new_train_data.update(generate_tests_probabilistically(noun_choice, ITERS, max_input_gen_threshold, mode, noun_error1, noun_error2, noun_dict1, noun_dict2))


# In[49]:


len(new_train_data)


# In[50]:


# dict(itertools.islice(new_train_data.items(), 2))


# ### D. Generate Data for Indirect Gender Bias, i.e. Name Bias

# In[51]:


noun_choice =  3 #Noun /Pronoun


# In[52]:


ITERS = 30000
num_iter = 5000 
max_input_gen_threshold =  3000
mode = "hub/gender-name-noun"


# In[53]:


#load pickles

pred_err_count, fairness_err_count = 0, 0

unique_pred_input1_error_set, unique_fairness_input1_error_set = set(), set() 
retrain_dict = dict()

count_dict = [{}, {}, {}, {}, {}, {}, {}, {}]
pred_error_dict = [{}, {}, {}, {}, {}, {}, {}, {}]
fairness_error_dict = [{}, {}, {}, {}, {}, {}, {}, {}]

bias_pair_count = [{}, {}, {}, {}, {}, {}]
bias_pair_pred_error = [{}, {}, {}, {}, {}, {}]
bias_pair_fairness_error = [{}, {}, {}, {}, {}, {}]
noun_dict1, noun_dict2, noun_error1, noun_error2 = {}, {}, {}, {}

c_vals = [noun_dict1, noun_dict2, noun_error1, noun_error2, unique_pred_input1_error_set, pred_err_count, unique_fairness_input1_error_set, fairness_err_count, retrain_dict, pred_error_dict, fairness_error_dict, bias_pair_pred_error, bias_pair_fairness_error]
c_names = ["noun_dict1", "noun_dict2", "noun_error1", "noun_error2", "unique_pred_input1_error_set", "pred_err_count", "unique_fairness_input1_error_set", "fairness_err_count", "retrain_dict", "pred_error_dict", "fairness_error_dict", "bias_pair_pred_error", "bias_pair_fairness_error"]

assert len(c_names) == len(c_vals),     "ERROR: bug in variables names for stored inputs {}, {}".format(c_names)

for i in range(0, len(c_vals)):
    store = c_names[i] + ".pickle"
    target = 'Exploitation/saved_pickles/exploitation/' + mode + '/' + store
    if os.path.exists(target):
        if os.path.getsize(target) > 0:
#             print(os.path.getsize(target))
#             print(os.path.exists(target))
            with open('Exploitation/saved_pickles/exploitation/' + mode + '/' + store, 'rb') as handle:
                c_vals[i] = pickle.load(handle)
                if i <4:
                    print(c_names[i], ": ", c_vals[i])
#         else:
#             print("ERROR {} is empty".format(target))
#     else:
#         print("ERROR {} does not exist".format(target))


# In[54]:


noun_dict1 = c_vals[0]
noun_dict2 = c_vals[1]
noun_error1 =  c_vals[2]
noun_error2 =  c_vals[3]

noun_dict1.pop('her', None)
noun_error1.pop('her', None)
noun_dict2.pop('him', None)
noun_error2.pop('him', None)


# In[55]:


print("noun_error1: {}".format(noun_error1))
print("noun_dict1: {}".format(noun_dict1))


print("noun_error2: {}".format(noun_error2))
print("noun_dict2: {}".format(noun_dict2))


# In[56]:


# generate new inputs probabilistically from 
new_train_data.update(generate_tests_probabilistically(noun_choice, ITERS, max_input_gen_threshold, mode, noun_error1, noun_error2, noun_dict1, noun_dict2))


# In[57]:


len(new_train_data)


# In[58]:


# dict(itertools.islice(new_train_data.items(), 2))


# ### E. Generate Data for Indirect Racial Bias, i.e. Name Bias

# In[59]:


noun_choice =  5 #Noun /Pronoun


# In[60]:


ITERS = 30000
num_iter = 5000 
max_input_gen_threshold =  3000
mode = "hub/racial-name-noun"


# In[61]:


#load pickles

pred_err_count, fairness_err_count = 0, 0

unique_pred_input1_error_set, unique_fairness_input1_error_set = set(), set() 
retrain_dict = dict()

count_dict = [{}, {}, {}, {}, {}, {}, {}, {}]
pred_error_dict = [{}, {}, {}, {}, {}, {}, {}, {}]
fairness_error_dict = [{}, {}, {}, {}, {}, {}, {}, {}]

bias_pair_count = [{}, {}, {}, {}, {}, {}]
bias_pair_pred_error = [{}, {}, {}, {}, {}, {}]
bias_pair_fairness_error = [{}, {}, {}, {}, {}, {}]
noun_dict1, noun_dict2, noun_error1, noun_error2 = {}, {}, {}, {}

c_vals = [noun_dict1, noun_dict2, noun_error1, noun_error2, unique_pred_input1_error_set, pred_err_count, unique_fairness_input1_error_set, fairness_err_count, retrain_dict, pred_error_dict, fairness_error_dict, bias_pair_pred_error, bias_pair_fairness_error]
c_names = ["noun_dict1", "noun_dict2", "noun_error1", "noun_error2", "unique_pred_input1_error_set", "pred_err_count", "unique_fairness_input1_error_set", "fairness_err_count", "retrain_dict", "pred_error_dict", "fairness_error_dict", "bias_pair_pred_error", "bias_pair_fairness_error"]

assert len(c_names) == len(c_vals),     "ERROR: bug in variables names for stored inputs {}, {}".format(c_names)

for i in range(0, len(c_vals)):
    store = c_names[i] + ".pickle"
    target = 'Exploitation/saved_pickles/exploitation/' + mode + '/' + store
    if os.path.exists(target):
        if os.path.getsize(target) > 0:
#             print(os.path.getsize(target))
#             print(os.path.exists(target))
            with open('Exploitation/saved_pickles/exploitation/' + mode + '/' + store, 'rb') as handle:
                c_vals[i] = pickle.load(handle)
                if i <4:
                    print(c_names[i], ": ", c_vals[i])
#         else:
#             print("ERROR {} is empty".format(target))
#     else:
#         print("ERROR {} does not exist".format(target))


# In[62]:


noun_dict1 = c_vals[0]
noun_dict2 = c_vals[1]
noun_error1 =  c_vals[2]
noun_error2 =  c_vals[3]

noun_dict1.pop('her', None)
noun_error1.pop('her', None)
noun_dict2.pop('him', None)
noun_error2.pop('him', None)


# In[63]:


print("noun_error1: {}".format(noun_error1))
print("noun_dict1: {}".format(noun_dict1))


print("noun_error2: {}".format(noun_error2))
print("noun_dict2: {}".format(noun_dict2))


# In[64]:


# generate new inputs probabilistically from 
new_train_data.update(generate_tests_probabilistically(noun_choice, ITERS, max_input_gen_threshold, mode, noun_error1, noun_error2, noun_dict1, noun_dict2))


# In[65]:


len(new_train_data)


# In[66]:


del noun_error2
del noun_dict2
del noun_dict1
del noun_error1
del c_vals
del pred_err_count
del fairness_err_count
del unique_pred_input1_error_set
del unique_fairness_input1_error_set
del retrain_dict
del count_dict


# In[67]:


del pred_error_dict
del fairness_error_dict
del bias_pair_count
del bias_pair_pred_error
del bias_pair_fairness_error
# del pred_err_count
# del fairness_err_count
# del unique_pred_input1_error_set
# del unique_fairness_input1_error_set
# del retrain_dict
# del count_dict


# In[ ]:





# In[ ]:





# ## Download the IMDB dataset
# 
# The IMDB dataset is available on [imdb reviews](https://www.tensorflow.org/datasets/catalog/imdb_reviews) or on [TensorFlow datasets](https://www.tensorflow.org/datasets). The following code downloads the IMDB dataset to your machine (or the colab runtime):

# In[68]:


# Split the training set into 60% and 40%, so we'll end up with 15,000 examples
# for training, 10,000 examples for validation and 25,000 examples for testing.
train_data, validation_data, test_data = tfds.load(
    name="imdb_reviews", 
    split=('train[:60%]', 'train[60%:]', 'test'),
    as_supervised=True)


# ### Augment IMDB Training Data with new generated test data 

# In[69]:


def slice_dict_randomly(new_train_data, size):
    res_dict = {}
    keys = random.sample(list(new_train_data), size)
#     values = [new_train_data[k] for k in keys]
    for i in keys:
        res_dict[i] = new_train_data[i]
    return res_dict


# In[70]:


print(slice_dict_randomly(new_train_data, 5))


# In[71]:


print(slice_dict_randomly(new_train_data, 5))


# In[72]:


print(slice_dict_randomly(new_train_data, 5))


# In[73]:


length = 15000


# In[74]:


# percent = [0.1 * len(train_data), 0.2 * len(train_data), 0.3 * len(train_data)]
percent = [0.01 * length, 0.02 * length, 0.03 * length, 0.04 * length, 0.05 * length, 0.06 * length, 0.07 * length, 0.08 * length, 0.09 * length, 0.1 * length]


# In[75]:


percent


# In[76]:


below_one_percent = [round(0.0025 * length, 0), round(0.005 * length, 0), round(0.0075 * length, 0)]


# In[77]:


below_one_percent


# In[78]:


for i in range(1, 11):
    str_var = "one_quarter_percent_dataset_rand_" +  str(i)
    locals()[str_var] = slice_dict_randomly(new_train_data, int(below_one_percent[0]))


# In[79]:


print(len(one_quarter_percent_dataset_rand_1))
print(len(one_quarter_percent_dataset_rand_10))


# In[80]:


one_quarter_percent_datasets = [one_quarter_percent_dataset_rand_1, one_quarter_percent_dataset_rand_2, one_quarter_percent_dataset_rand_3, one_quarter_percent_dataset_rand_4,                             one_quarter_percent_dataset_rand_5, one_quarter_percent_dataset_rand_6, one_quarter_percent_dataset_rand_7, one_quarter_percent_dataset_rand_8,                             one_quarter_percent_dataset_rand_9, one_quarter_percent_dataset_rand_10]


# In[81]:


for i in range(1, 11):
    str_var = "half_percent_dataset_rand_" +  str(i)
    locals()[str_var] = slice_dict_randomly(new_train_data, int(below_one_percent[1]))


# In[82]:


half_percent_datasets = [half_percent_dataset_rand_1, half_percent_dataset_rand_2, half_percent_dataset_rand_3, half_percent_dataset_rand_4,                             half_percent_dataset_rand_5, half_percent_dataset_rand_6, half_percent_dataset_rand_7, half_percent_dataset_rand_8,                             half_percent_dataset_rand_9, half_percent_dataset_rand_10]


# In[83]:


for i in range(1, 11):
    str_var = "three_quarter_percent_dataset_rand_" +  str(i)
    locals()[str_var] = slice_dict_randomly(new_train_data, int(below_one_percent[2]))


# In[84]:


three_quarter_percent_datasets = [three_quarter_percent_dataset_rand_1, three_quarter_percent_dataset_rand_2, three_quarter_percent_dataset_rand_3, three_quarter_percent_dataset_rand_4,                             three_quarter_percent_dataset_rand_5, three_quarter_percent_dataset_rand_6, three_quarter_percent_dataset_rand_7, three_quarter_percent_dataset_rand_8,                             three_quarter_percent_dataset_rand_9, three_quarter_percent_dataset_rand_10]


# In[85]:


len(one_quarter_percent_dataset_rand_1)


# In[86]:


len(half_percent_dataset_rand_1)


# In[87]:


len(three_quarter_percent_dataset_rand_1)


# In[88]:


for i in range(1, 11):
    str_var = "one_percent_dataset_rand_" +  str(i)
    locals()[str_var] = slice_dict_randomly(new_train_data, int(percent[0]))


# In[89]:


one_percent_datasets = [one_percent_dataset_rand_1, one_percent_dataset_rand_2, one_percent_dataset_rand_3, one_percent_dataset_rand_4,                             one_percent_dataset_rand_5, one_percent_dataset_rand_6, one_percent_dataset_rand_7, one_percent_dataset_rand_8,                             one_percent_dataset_rand_9, one_percent_dataset_rand_10]


# In[90]:


for i in range(1, 11):
    str_var = "two_percent_dataset_rand_" +  str(i)
    locals()[str_var] = slice_dict_randomly(new_train_data, int(percent[1]))


# In[91]:


two_percent_datasets = [two_percent_dataset_rand_1, two_percent_dataset_rand_2, two_percent_dataset_rand_3, two_percent_dataset_rand_4,                             two_percent_dataset_rand_5, two_percent_dataset_rand_6, two_percent_dataset_rand_7, two_percent_dataset_rand_8,                             two_percent_dataset_rand_9, two_percent_dataset_rand_10]


# In[92]:


for i in range(1, 11):
    str_var = "three_percent_dataset_rand_" +  str(i)
    locals()[str_var] = slice_dict_randomly(new_train_data, int(percent[2]))


# In[93]:


three_percent_datasets = [three_percent_dataset_rand_1, three_percent_dataset_rand_2, three_percent_dataset_rand_3, three_percent_dataset_rand_4,                             three_percent_dataset_rand_5, three_percent_dataset_rand_6, three_percent_dataset_rand_7, three_percent_dataset_rand_8,                             three_percent_dataset_rand_9, three_percent_dataset_rand_10]


# In[94]:


for i in range(1, 11):
    str_var = "four_percent_dataset_rand_" +  str(i)
    locals()[str_var] = slice_dict_randomly(new_train_data, int(percent[3]))


# In[95]:


four_percent_datasets = [four_percent_dataset_rand_1, four_percent_dataset_rand_2, four_percent_dataset_rand_3, four_percent_dataset_rand_4,                             four_percent_dataset_rand_5, four_percent_dataset_rand_6, four_percent_dataset_rand_7, four_percent_dataset_rand_8,                             four_percent_dataset_rand_9, four_percent_dataset_rand_10]


# In[96]:


for i in range(1, 11):
    str_var = "five_percent_dataset_rand_" +  str(i)
    locals()[str_var] = slice_dict_randomly(new_train_data, int(percent[4]))


# In[97]:


five_percent_datasets = [five_percent_dataset_rand_1, five_percent_dataset_rand_2, five_percent_dataset_rand_3, five_percent_dataset_rand_4,                             five_percent_dataset_rand_5, five_percent_dataset_rand_6, five_percent_dataset_rand_7, five_percent_dataset_rand_8,                             five_percent_dataset_rand_9, five_percent_dataset_rand_10]


# In[98]:


for i in range(1, 11):
    str_var = "six_percent_dataset_rand_" +  str(i)
    locals()[str_var] = slice_dict_randomly(new_train_data, int(percent[5]))


# In[99]:


six_percent_datasets = [six_percent_dataset_rand_1, six_percent_dataset_rand_2, six_percent_dataset_rand_3, six_percent_dataset_rand_4,                             six_percent_dataset_rand_5, six_percent_dataset_rand_6, six_percent_dataset_rand_7, six_percent_dataset_rand_8,                             six_percent_dataset_rand_9, six_percent_dataset_rand_10]


# In[100]:


for i in range(1, 11):
    str_var = "seven_percent_dataset_rand_" +  str(i)
    locals()[str_var] = slice_dict_randomly(new_train_data, int(percent[6]))


# In[101]:


seven_percent_datasets = [seven_percent_dataset_rand_1, seven_percent_dataset_rand_2, seven_percent_dataset_rand_3, seven_percent_dataset_rand_4,                             seven_percent_dataset_rand_5, seven_percent_dataset_rand_6, seven_percent_dataset_rand_7, seven_percent_dataset_rand_8,                             seven_percent_dataset_rand_9, seven_percent_dataset_rand_10]


# In[102]:


for i in range(1, 11):
    str_var = "eight_percent_dataset_rand_" +  str(i)
    locals()[str_var] = slice_dict_randomly(new_train_data, int(percent[7]))


# In[103]:


eight_percent_datasets = [eight_percent_dataset_rand_1, eight_percent_dataset_rand_2, eight_percent_dataset_rand_3, eight_percent_dataset_rand_4,                             eight_percent_dataset_rand_5, eight_percent_dataset_rand_6, eight_percent_dataset_rand_7, eight_percent_dataset_rand_8,                             eight_percent_dataset_rand_9, eight_percent_dataset_rand_10]


# In[104]:


for i in range(1, 11):
    str_var = "nine_percent_dataset_rand_" +  str(i)
    locals()[str_var] = slice_dict_randomly(new_train_data, int(percent[8]))


# In[105]:


nine_percent_datasets = [nine_percent_dataset_rand_1, nine_percent_dataset_rand_2, nine_percent_dataset_rand_3, nine_percent_dataset_rand_4,                             nine_percent_dataset_rand_5, nine_percent_dataset_rand_6, nine_percent_dataset_rand_7, nine_percent_dataset_rand_8,                             nine_percent_dataset_rand_9, nine_percent_dataset_rand_10]


# In[106]:


for i in range(1, 11):
    str_var = "ten_percent_dataset_rand_" +  str(i)
    locals()[str_var] = slice_dict_randomly(new_train_data, int(percent[9]))


# In[107]:


ten_percent_datasets = [ten_percent_dataset_rand_1, ten_percent_dataset_rand_2, ten_percent_dataset_rand_3, ten_percent_dataset_rand_4,                             ten_percent_dataset_rand_5, ten_percent_dataset_rand_6, ten_percent_dataset_rand_7, ten_percent_dataset_rand_8,                             ten_percent_dataset_rand_9, ten_percent_dataset_rand_10]


# In[108]:


len(ten_percent_dataset_rand_1)


# In[109]:


len(ten_percent_dataset_rand_10)


# In[110]:


len(ten_percent_datasets)


# In[111]:


import gc
gc.collect()


# In[112]:


len(ten_percent_datasets)


# In[113]:


# from tensorflow.python.framework import ops
# ops.reset_default_graph()


# In[114]:


def add_new_data(percent_dataset, train_data):
#     res_data = train_data.prefetch(1)
#     print("Init res_data: ", len(res_data))
#     print("Type res_data: ", type(res_data))
#     res_data.clear()
#     prefetched_tensor_tuple = tuple()
    c = 0
    for i in percent_dataset:
        c+= 1
#         if (c % 2) == 0:
#             gc.collect()
        label_tmp = tf.constant([percent_dataset[i]]).numpy()[0]
        data_tmp = tf.constant([i]).numpy()[0]
        label = tf.convert_to_tensor(label_tmp, np.int64)
        data = tf.convert_to_tensor(data_tmp)
        data_point = (data, label)
        tensor_data_point = tf.data.Dataset.from_tensors(data_point)
        prefetched_tensor = tensor_data_point.prefetch(len(tensor_data_point))
#         prefetched_tensor = tensor_data_point.prefetch(1)
#         prefetched_tensor_tuple
        train_data = train_data.concatenate(prefetched_tensor) 
#         ops.reset_default_graph()
#     print("Final res_data: ", len(res_data))
#     print("Type res_data: ", type(res_data))
#         tf.reset_default_graph #()
#     tf.reset_default_graph #()
#     ops.reset_default_graph()
    return train_data #.concatenate(res_data)


# In[115]:


len(train_data)


# In[116]:


i = 0
for dataset in one_quarter_percent_datasets:
    i += 1 
    str_var = "one_quarter_percent_additional_train_dataset_rand_" + str(i) 
    locals()[str_var]= add_new_data(dataset, train_data)


# In[117]:


print(len(one_quarter_percent_additional_train_dataset_rand_1))
print(len(one_quarter_percent_additional_train_dataset_rand_10))


# In[118]:


i = 0
for dataset in half_percent_datasets:
    i += 1 
    str_var = "half_percent_additional_train_dataset_rand_" + str(i) 
    locals()[str_var]= add_new_data(dataset, train_data)


# In[119]:


print(len(half_percent_additional_train_dataset_rand_1))
print(len(half_percent_additional_train_dataset_rand_10))


# In[120]:


i = 0
for dataset in three_quarter_percent_datasets:
    i += 1 
    str_var = "three_quarter_percent_additional_train_dataset_rand_" + str(i) 
    locals()[str_var]= add_new_data(dataset, train_data)


# In[121]:


print(len(three_quarter_percent_additional_train_dataset_rand_1))
print(len(three_quarter_percent_additional_train_dataset_rand_10))


# In[122]:


i = 0
for dataset in one_percent_datasets:
    i += 1 
    str_var = "one_percent_additional_train_dataset_rand_" + str(i) 
    locals()[str_var]= add_new_data(dataset, train_data)


# In[123]:


print(len(one_percent_additional_train_dataset_rand_1))
print(len(one_percent_additional_train_dataset_rand_10))


# In[124]:


i = 0
for dataset in two_percent_datasets:
    i += 1 
    str_var = "two_percent_additional_train_dataset_rand_" + str(i) 
    locals()[str_var]= add_new_data(dataset, train_data)


# In[125]:


print(len(two_percent_additional_train_dataset_rand_1))
print(len(two_percent_additional_train_dataset_rand_10))


# In[126]:


i = 0
for dataset in three_percent_datasets:
    i += 1 
    str_var = "three_percent_additional_train_dataset_rand_" + str(i) 
    locals()[str_var]= add_new_data(dataset, train_data)


# In[127]:


print(len(three_percent_additional_train_dataset_rand_1))
print(len(three_percent_additional_train_dataset_rand_10))


# In[128]:


i = 0
for dataset in four_percent_datasets:
    i += 1 
    str_var = "four_percent_additional_train_dataset_rand_" + str(i) 
    locals()[str_var]= add_new_data(dataset, train_data)


# In[129]:


print(len(four_percent_additional_train_dataset_rand_1))
print(len(four_percent_additional_train_dataset_rand_10))


# In[130]:


i = 0
for dataset in five_percent_datasets:
    i += 1 
    str_var = "five_percent_additional_train_dataset_rand_" + str(i) 
    locals()[str_var]= add_new_data(dataset, train_data)


# In[131]:


print(len(five_percent_additional_train_dataset_rand_1))
print(len(five_percent_additional_train_dataset_rand_10))


# In[132]:


i = 0
for dataset in six_percent_datasets:
    i += 1 
    str_var = "six_percent_additional_train_dataset_rand_" + str(i) 
    locals()[str_var]= add_new_data(dataset, train_data)


# In[133]:


print(len(six_percent_additional_train_dataset_rand_1))
print(len(six_percent_additional_train_dataset_rand_10))


# In[134]:


i = 0
for dataset in seven_percent_datasets:
    i += 1 
    str_var = "seven_percent_additional_train_dataset_rand_" + str(i) 
    locals()[str_var]= add_new_data(dataset, train_data)


# In[135]:


print(len(seven_percent_additional_train_dataset_rand_1))
print(len(seven_percent_additional_train_dataset_rand_10))


# In[136]:


i = 0
for dataset in eight_percent_datasets:
    i += 1 
    str_var = "eight_percent_additional_train_dataset_rand_" + str(i) 
    locals()[str_var]= add_new_data(dataset, train_data)


# In[137]:


print(len(eight_percent_additional_train_dataset_rand_1))
print(len(eight_percent_additional_train_dataset_rand_10))


# In[138]:


i = 0
for dataset in nine_percent_datasets:
    i += 1 
    str_var = "nine_percent_additional_train_dataset_rand_" + str(i) 
    locals()[str_var]= add_new_data(dataset, train_data)


# In[139]:


print(len(nine_percent_additional_train_dataset_rand_1))
print(len(nine_percent_additional_train_dataset_rand_10))


# In[140]:


i = 0
for dataset in ten_percent_datasets:
    i += 1 
    str_var = "ten_percent_additional_train_dataset_rand_" + str(i) 
    locals()[str_var]= add_new_data(dataset, train_data)


# In[141]:


print(len(ten_percent_additional_train_dataset_rand_1))
print(len(ten_percent_additional_train_dataset_rand_10))


# In[142]:


one_quarter_percent_additional_train_datasets = [one_quarter_percent_additional_train_dataset_rand_1, one_quarter_percent_additional_train_dataset_rand_2,                                         one_quarter_percent_additional_train_dataset_rand_3, one_quarter_percent_additional_train_dataset_rand_4,                                         one_quarter_percent_additional_train_dataset_rand_5, one_quarter_percent_additional_train_dataset_rand_6,                                         one_quarter_percent_additional_train_dataset_rand_7, one_quarter_percent_additional_train_dataset_rand_8,                                         one_quarter_percent_additional_train_dataset_rand_9, one_quarter_percent_additional_train_dataset_rand_10]


# In[143]:


half_percent_additional_train_datasets = [half_percent_additional_train_dataset_rand_1, half_percent_additional_train_dataset_rand_2,                                         half_percent_additional_train_dataset_rand_3, half_percent_additional_train_dataset_rand_4,                                         half_percent_additional_train_dataset_rand_5, half_percent_additional_train_dataset_rand_6,                                         half_percent_additional_train_dataset_rand_7, half_percent_additional_train_dataset_rand_8,                                         half_percent_additional_train_dataset_rand_9, half_percent_additional_train_dataset_rand_10]


# In[144]:


three_quarter_percent_additional_train_datasets = [three_quarter_percent_additional_train_dataset_rand_1, three_quarter_percent_additional_train_dataset_rand_2,                                         three_quarter_percent_additional_train_dataset_rand_3, three_quarter_percent_additional_train_dataset_rand_4,                                         three_quarter_percent_additional_train_dataset_rand_5, three_quarter_percent_additional_train_dataset_rand_6,                                         three_quarter_percent_additional_train_dataset_rand_7, three_quarter_percent_additional_train_dataset_rand_8,                                         three_quarter_percent_additional_train_dataset_rand_9, three_quarter_percent_additional_train_dataset_rand_10]


# In[145]:


one_percent_additional_train_datasets = [one_percent_additional_train_dataset_rand_1, one_percent_additional_train_dataset_rand_2,                                         one_percent_additional_train_dataset_rand_3, one_percent_additional_train_dataset_rand_4,                                         one_percent_additional_train_dataset_rand_5, one_percent_additional_train_dataset_rand_6,                                         one_percent_additional_train_dataset_rand_7, one_percent_additional_train_dataset_rand_8,                                         one_percent_additional_train_dataset_rand_9, one_percent_additional_train_dataset_rand_10]


# In[146]:


len(one_percent_additional_train_datasets)


# In[147]:


two_percent_additional_train_datasets = [two_percent_additional_train_dataset_rand_1, two_percent_additional_train_dataset_rand_2,                                         two_percent_additional_train_dataset_rand_3, two_percent_additional_train_dataset_rand_4,                                         two_percent_additional_train_dataset_rand_5, two_percent_additional_train_dataset_rand_6,                                         two_percent_additional_train_dataset_rand_7, two_percent_additional_train_dataset_rand_8,                                         two_percent_additional_train_dataset_rand_9, two_percent_additional_train_dataset_rand_10]


# In[148]:


three_percent_additional_train_datasets = [three_percent_additional_train_dataset_rand_1, three_percent_additional_train_dataset_rand_2,                                         three_percent_additional_train_dataset_rand_3, three_percent_additional_train_dataset_rand_4,                                         three_percent_additional_train_dataset_rand_5, three_percent_additional_train_dataset_rand_6,                                         three_percent_additional_train_dataset_rand_7, three_percent_additional_train_dataset_rand_8,                                         three_percent_additional_train_dataset_rand_9, three_percent_additional_train_dataset_rand_10]


# In[149]:


four_percent_additional_train_datasets = [four_percent_additional_train_dataset_rand_1, four_percent_additional_train_dataset_rand_2,                                         four_percent_additional_train_dataset_rand_3, four_percent_additional_train_dataset_rand_4,                                         four_percent_additional_train_dataset_rand_5, four_percent_additional_train_dataset_rand_6,                                         four_percent_additional_train_dataset_rand_7, four_percent_additional_train_dataset_rand_8,                                         four_percent_additional_train_dataset_rand_9, four_percent_additional_train_dataset_rand_10]


# In[150]:


five_percent_additional_train_datasets = [five_percent_additional_train_dataset_rand_1, five_percent_additional_train_dataset_rand_2,                                         five_percent_additional_train_dataset_rand_3, five_percent_additional_train_dataset_rand_4,                                         five_percent_additional_train_dataset_rand_5, five_percent_additional_train_dataset_rand_6,                                         five_percent_additional_train_dataset_rand_7, five_percent_additional_train_dataset_rand_8,                                         five_percent_additional_train_dataset_rand_9, five_percent_additional_train_dataset_rand_10]


# In[151]:


six_percent_additional_train_datasets = [six_percent_additional_train_dataset_rand_1, six_percent_additional_train_dataset_rand_2,                                         six_percent_additional_train_dataset_rand_3, six_percent_additional_train_dataset_rand_4,                                         six_percent_additional_train_dataset_rand_5, six_percent_additional_train_dataset_rand_6,                                         six_percent_additional_train_dataset_rand_7, six_percent_additional_train_dataset_rand_8,                                         six_percent_additional_train_dataset_rand_9, six_percent_additional_train_dataset_rand_10]


# In[152]:


seven_percent_additional_train_datasets = [seven_percent_additional_train_dataset_rand_1, seven_percent_additional_train_dataset_rand_2,                                         seven_percent_additional_train_dataset_rand_3, seven_percent_additional_train_dataset_rand_4,                                         seven_percent_additional_train_dataset_rand_5, seven_percent_additional_train_dataset_rand_6,                                         seven_percent_additional_train_dataset_rand_7, seven_percent_additional_train_dataset_rand_8,                                         seven_percent_additional_train_dataset_rand_9, seven_percent_additional_train_dataset_rand_10]


# In[153]:


eight_percent_additional_train_datasets = [eight_percent_additional_train_dataset_rand_1, eight_percent_additional_train_dataset_rand_2,                                         eight_percent_additional_train_dataset_rand_3, eight_percent_additional_train_dataset_rand_4,                                         eight_percent_additional_train_dataset_rand_5, eight_percent_additional_train_dataset_rand_6,                                         eight_percent_additional_train_dataset_rand_7, eight_percent_additional_train_dataset_rand_8,                                         eight_percent_additional_train_dataset_rand_9, eight_percent_additional_train_dataset_rand_10]


# In[154]:


nine_percent_additional_train_datasets = [nine_percent_additional_train_dataset_rand_1, nine_percent_additional_train_dataset_rand_2,                                         nine_percent_additional_train_dataset_rand_3, nine_percent_additional_train_dataset_rand_4,                                         nine_percent_additional_train_dataset_rand_5, nine_percent_additional_train_dataset_rand_6,                                         nine_percent_additional_train_dataset_rand_7, nine_percent_additional_train_dataset_rand_8,                                         nine_percent_additional_train_dataset_rand_9, nine_percent_additional_train_dataset_rand_10]


# In[155]:


ten_percent_additional_train_datasets = [ten_percent_additional_train_dataset_rand_1, ten_percent_additional_train_dataset_rand_2,                                         ten_percent_additional_train_dataset_rand_3, ten_percent_additional_train_dataset_rand_4,                                         ten_percent_additional_train_dataset_rand_5, ten_percent_additional_train_dataset_rand_6,                                         ten_percent_additional_train_dataset_rand_7, ten_percent_additional_train_dataset_rand_8,                                         ten_percent_additional_train_dataset_rand_9, ten_percent_additional_train_dataset_rand_10]


# In[156]:


def print_at_thresholds(dataset):
    t1 = 15000 #int(percent[0])
    t2 = len(dataset)
    print("len: ", t2)
    print(" " * 30)
    c = 0
    for i in dataset:
        if c < 2:
            print("data @ {} is {}".format(c, i))
            print(" " * 30)
        if (c == t1) or (c == (t1-1)):
            print("data @ {} is {}".format(c, i))
            print(" " * 30)
        if (c == (t2-1)) or (c == (t2-2)):
            print("*" * 25, "NEW")
            print("data @ {} is {}".format(c, i))
            print(" " * 30)
        c += 1


# In[157]:


print_at_thresholds(one_percent_additional_train_dataset_rand_1)


# In[158]:


print_at_thresholds(two_percent_additional_train_dataset_rand_1)


# In[159]:


print_at_thresholds(three_percent_additional_train_dataset_rand_1)


# In[160]:


print_at_thresholds(four_percent_additional_train_dataset_rand_1)


# In[161]:


print_at_thresholds(five_percent_additional_train_dataset_rand_1)


# In[162]:


print_at_thresholds(six_percent_additional_train_dataset_rand_1)


# In[163]:


print_at_thresholds(eight_percent_additional_train_dataset_rand_1)


# In[164]:


gc.collect()


# In[165]:


# automatically create directories for saving models
def create_model_subdir(dataset):
    s = dataset.split("percent")
    target_dir = 'saved_model/' + s[0] + "percent_models"
    if not os.path.exists(os.path.join(os.getcwd(), target_dir)):
        sub_dir = target_dir.split("/")
        k = os.getcwd()
        for dir_loc in sub_dir:
            k = os.path.join(k, dir_loc)
            if not os.path.exists(str(k)):
                os.mkdir(k)


# ## Explore the data 
# 
# Let's take a moment to understand the format of the data. Each example is a sentence representing the movie review and a corresponding label. The sentence is not preprocessed in any way. The label is an integer value of either 0 or 1, where 0 is a negative review, and 1 is a positive review.
# 
# Let's print first 10 examples.

# In[166]:


# train_examples_batch, train_labels_batch = next(iter(one_percent_additional_train_dataset_rand_1.batch(10)))
# train_examples_batch


# Let's also print the first 10 labels.

# In[167]:


# train_labels_batch


# ## Build the model
# 
# The neural network is created by stacking layers—this requires three main architectural decisions:
# 
# * How to represent the text?
# * How many layers to use in the model?
# * How many *hidden units* to use for each layer?
# 
# In this example, the input data consists of sentences. The labels to predict are either 0 or 1.
# 
# One way to represent the text is to convert sentences into embeddings vectors. We can use a pre-trained text embedding as the first layer, which will have three advantages:
# 
# *   we don't have to worry about text preprocessing,
# *   we can benefit from transfer learning,
# *   the embedding has a fixed size, so it's simpler to process.
# 
# For this example we will use a **pre-trained text embedding model** from [TensorFlow Hub](https://www.tensorflow.org/hub) called [google/tf2-preview/gnews-swivel-20dim/1](https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1).
# 
# There are three other pre-trained models to test for the sake of this tutorial:
# 
# * [google/tf2-preview/gnews-swivel-20dim-with-oov/1](https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim-with-oov/1) - same as [google/tf2-preview/gnews-swivel-20dim/1](https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1), but with 2.5% vocabulary converted to OOV buckets. This can help if vocabulary of the task and vocabulary of the model don't fully overlap.
# * [google/tf2-preview/nnlm-en-dim50/1](https://tfhub.dev/google/tf2-preview/nnlm-en-dim50/1) - A much larger model with ~1M vocabulary size and 50 dimensions.
# * [google/tf2-preview/nnlm-en-dim128/1](https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1) - Even larger model with ~1M vocabulary size and 128 dimensions.

# Let's first create a Keras layer that uses a TensorFlow Hub model to embed the sentences, and try it out on a couple of input examples. Note that no matter the length of the input text, the output shape of the embeddings is: `(num_examples, embedding_dimension)`.

# In[168]:


# embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
# hub_layer = hub.KerasLayer(embedding, input_shape=[], 
#                            dtype=tf.string, trainable=True)
# hub_layer(train_examples_batch[:3])


# Let's now build the full model:

# In[169]:


# model = tf.keras.Sequential()
# model.add(hub_layer)
# model.add(tf.keras.layers.Dense(16, activation='relu'))
# model.add(tf.keras.layers.Dense(1))

# model.summary()


# The layers are stacked sequentially to build the classifier:
# 
# 1. The first layer is a TensorFlow Hub layer. This layer uses a pre-trained Saved Model to map a sentence into its embedding vector. The pre-trained text embedding model that we are using ([google/tf2-preview/gnews-swivel-20dim/1](https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1)) splits the sentence into tokens, embeds each token and then combines the embedding. The resulting dimensions are: `(num_examples, embedding_dimension)`.
# 2. This fixed-length output vector is piped through a fully-connected (`Dense`) layer with 16 hidden units.
# 3. The last layer is densely connected with a single output node.
# 
# Let's compile the model.

# ### Loss function and optimizer
# 
# A model needs a loss function and an optimizer for training. Since this is a binary classification problem and the model outputs logits (a single-unit layer with a linear activation), we'll use the `binary_crossentropy` loss function.
# 
# This isn't the only choice for a loss function, you could, for instance, choose `mean_squared_error`. But, generally, `binary_crossentropy` is better for dealing with probabilities—it measures the "distance" between probability distributions, or in our case, between the ground-truth distribution and the predictions.
# 
# Later, when we are exploring regression problems (say, to predict the price of a house), we will see how to use another loss function called mean squared error.
# 
# Now, configure the model to use an optimizer and a loss function:

# In[170]:


# model.compile(optimizer='adam',
#               loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#               metrics=['accuracy'])


# In[171]:


# train_data = ten_percent_additional_train_dataset


# In[172]:


# train_data.shuffle(5000)


# In[ ]:





# ## Train 10 models with 0.25% of new training data

# In[173]:


# import tensorflow as tf
# import gc

# gc.collect()

# class MyCustomCallback(tf.keras.callbacks.Callback):
#   def on_epoch_end(self, epoch, logs=None):
#     gc.collect()


# In[174]:


# i = 0
# datasets ="one_quarter_percent_additional_train_datasets"
# create_model_subdir(datasets)
# s = datasets.split("percent")
# pth = s[0] + "percent_models"

# for dataset in one_quarter_percent_additional_train_datasets:
#     i +=1 
    
#     train_examples_batch, train_labels_batch = next(iter(dataset.batch(10)))
    
#     embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
#     hub_layer = hub.KerasLayer(embedding, input_shape=[], 
#                                dtype=tf.string, trainable=True)
#     hub_layer(train_examples_batch[:3])

#     model = tf.keras.Sequential()
#     model.add(hub_layer)
#     model.add(tf.keras.layers.Dense(16, activation='relu'))
#     model.add(tf.keras.layers.Dense(1))

#     model.summary()
    
    
#     model.compile(optimizer='adam',
#               loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#               metrics=['accuracy'])
    
#     history = model.fit(dataset.shuffle(10000).batch(512),
#                     epochs=20,
#                     validation_data=validation_data.batch(512),
#                     verbose=0) #, use_multiprocessing=True)
    
#     results = model.evaluate(test_data.batch(512), verbose=0)

#     print("RESULTS for dataset - {} with rand id #: {}".format(datasets, str(i)))
#     for name, value in zip(model.metrics_names, results):
#       print("%s: %.3f" % (name, value))
    
#     model_path = "saved_model/" + pth + "/my_model_txt_classifier_hub_with_one_quarter_percent_extra_data" + "_rand_" + str(i)
#     model.save(model_path)


# ## Train 10 models with 0.5% of new training data

# In[175]:


# gc.collect()

# class MyCustomCallback(tf.keras.callbacks.Callback):
#   def on_epoch_end(self, epoch, logs=None):
#     gc.collect()


# In[176]:


# i = 0
# datasets ="half_percent_additional_train_datasets"
# create_model_subdir(datasets)
# s = datasets.split("percent")
# pth = s[0] + "percent_models"

# for dataset in half_percent_additional_train_datasets:
#     i +=1 
    
#     train_examples_batch, train_labels_batch = next(iter(dataset.batch(10)))
       
#     embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
#     hub_layer = hub.KerasLayer(embedding, input_shape=[], 
#                                dtype=tf.string, trainable=True)
#     hub_layer(train_examples_batch[:3])

#     model = tf.keras.Sequential()
#     model.add(hub_layer)
#     model.add(tf.keras.layers.Dense(16, activation='relu'))
#     model.add(tf.keras.layers.Dense(1))

#     model.summary()
    
    
#     model.compile(optimizer='adam',
#               loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#               metrics=['accuracy'])
    
#     history = model.fit(dataset.shuffle(10000).batch(512),
#                     epochs=20,
#                     validation_data=validation_data.batch(512),
#                     verbose=0) #, use_multiprocessing=True)
    
#     results = model.evaluate(test_data.batch(512), verbose=0)

#     print("RESULTS for dataset - {} with rand id #: {}".format(datasets, str(i)))
#     for name, value in zip(model.metrics_names, results):
#       print("%s: %.3f" % (name, value))
    
#     model_path = "saved_model/" + pth + "/my_model_txt_classifier_hub_with_half_percent_extra_data" + "_rand_" + str(i)
#     model.save(model_path)


# ## Train 10 models with 0.75% of new training data

# In[177]:


# gc.collect()

# class MyCustomCallback(tf.keras.callbacks.Callback):
#   def on_epoch_end(self, epoch, logs=None):
#     gc.collect()


# In[178]:


# i = 0
# datasets ="three_quarter_percent_additional_train_datasets"
# create_model_subdir(datasets)
# s = datasets.split("percent")
# pth = s[0] + "percent_models"

# for dataset in three_quarter_percent_additional_train_datasets:
#     i +=1 
    
#     train_examples_batch, train_labels_batch = next(iter(dataset.batch(10)))
    
#     embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
#     hub_layer = hub.KerasLayer(embedding, input_shape=[], 
#                                dtype=tf.string, trainable=True)
#     hub_layer(train_examples_batch[:3])

#     model = tf.keras.Sequential()
#     model.add(hub_layer)
#     model.add(tf.keras.layers.Dense(16, activation='relu'))
#     model.add(tf.keras.layers.Dense(1))

#     model.summary()
    
    
#     model.compile(optimizer='adam',
#               loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#               metrics=['accuracy'])
    
#     history = model.fit(dataset.shuffle(10000).batch(512),
#                     epochs=20,
#                     validation_data=validation_data.batch(512),
#                     verbose=0) #, use_multiprocessing=True)
    
#     results = model.evaluate(test_data.batch(512), verbose=0)

#     print("RESULTS for dataset - {} with rand id #: {}".format(datasets, str(i)))
#     for name, value in zip(model.metrics_names, results):
#       print("%s: %.3f" % (name, value))
    
#     model_path = "saved_model/" + pth + "/my_model_txt_classifier_hub_with_three_quarter_percent_extra_data" + "_rand_" + str(i)
#     model.save(model_path)


# ## Train 10 models with 1% of new training data

# In[179]:



# gc.collect()

# class MyCustomCallback(tf.keras.callbacks.Callback):
#   def on_epoch_end(self, epoch, logs=None):
#     gc.collect()


# In[180]:


# i = 0
# datasets ="one_percent_additional_train_datasets"
# create_model_subdir(datasets)
# s = datasets.split("percent")
# pth = s[0] + "percent_models"

# for dataset in one_percent_additional_train_datasets:
#     i +=1 
    
#     train_examples_batch, train_labels_batch = next(iter(dataset.batch(10)))
    
#     embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
#     hub_layer = hub.KerasLayer(embedding, input_shape=[], 
#                                dtype=tf.string, trainable=True)
#     hub_layer(train_examples_batch[:3])

#     model = tf.keras.Sequential()
#     model.add(hub_layer)
#     model.add(tf.keras.layers.Dense(16, activation='relu'))
#     model.add(tf.keras.layers.Dense(1))

#     model.summary()
    
    
#     model.compile(optimizer='adam',
#               loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#               metrics=['accuracy'])
    
#     history = model.fit(dataset.shuffle(10000).batch(512),
#                     epochs=20,
#                     validation_data=validation_data.batch(512),
#                     verbose=0) #, use_multiprocessing=True)
    
#     results = model.evaluate(test_data.batch(512), verbose=0)

#     print("RESULTS for dataset - {} with rand id #: {}".format(datasets, str(i)))
#     for name, value in zip(model.metrics_names, results):
#       print("%s: %.3f" % (name, value))
    
#     model_path = "saved_model/" + pth + "/my_model_txt_classifier_hub_with_one_percent_extra_data" + "_rand_" + str(i)
#     model.save(model_path)


# ## Train 10 models with 2% of new training data

# In[181]:



# gc.collect()

# class MyCustomCallback(tf.keras.callbacks.Callback):
#   def on_epoch_end(self, epoch, logs=None):
#     gc.collect()


# In[182]:


# i = 0
# datasets ="two_percent_additional_train_datasets"
# create_model_subdir(datasets)
# s = datasets.split("percent")
# pth = s[0] + "percent_models"

# for dataset in two_percent_additional_train_datasets:
#     i +=1 
    
#     train_examples_batch, train_labels_batch = next(iter(dataset.batch(10)))
    
#     embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
#     hub_layer = hub.KerasLayer(embedding, input_shape=[], 
#                                dtype=tf.string, trainable=True)
#     hub_layer(train_examples_batch[:3])

#     model = tf.keras.Sequential()
#     model.add(hub_layer)
#     model.add(tf.keras.layers.Dense(16, activation='relu'))
#     model.add(tf.keras.layers.Dense(1))

#     model.summary()
    
    
#     model.compile(optimizer='adam',
#               loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#               metrics=['accuracy'])
    
#     history = model.fit(dataset.shuffle(10000).batch(512),
#                     epochs=20,
#                     validation_data=validation_data.batch(512),
#                     verbose=0) #, use_multiprocessing=True)
    
#     results = model.evaluate(test_data.batch(512), verbose=0)

#     print("RESULTS for dataset - {} with rand id #: {}".format(datasets, str(i)))
#     for name, value in zip(model.metrics_names, results):
#       print("%s: %.3f" % (name, value))
    
#     model_path = "saved_model/" + pth + "/my_model_txt_classifier_hub_with_two_percent_extra_data" + "_rand_" + str(i)
#     model.save(model_path)


# ## Train 10 models with 3% of new training data

# In[183]:



# gc.collect()

# class MyCustomCallback(tf.keras.callbacks.Callback):
#   def on_epoch_end(self, epoch, logs=None):
#     gc.collect()


# In[184]:


# i = 0
# datasets ="three_percent_additional_train_datasets"
# create_model_subdir(datasets)
# s = datasets.split("percent")
# pth = s[0] + "percent_models"

# for dataset in three_percent_additional_train_datasets:
#     i +=1 
    
#     train_examples_batch, train_labels_batch = next(iter(dataset.batch(10)))
    
#     embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
#     hub_layer = hub.KerasLayer(embedding, input_shape=[], 
#                                dtype=tf.string, trainable=True)
#     hub_layer(train_examples_batch[:3])

#     model = tf.keras.Sequential()
#     model.add(hub_layer)
#     model.add(tf.keras.layers.Dense(16, activation='relu'))
#     model.add(tf.keras.layers.Dense(1))

#     model.summary()
    
    
#     model.compile(optimizer='adam',
#               loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#               metrics=['accuracy'])
    
#     history = model.fit(dataset.shuffle(10000).batch(512),
#                     epochs=20,
#                     validation_data=validation_data.batch(512),
#                     verbose=0) #, use_multiprocessing=True)
    
#     results = model.evaluate(test_data.batch(512), verbose=0)

#     print("RESULTS for dataset - {} with rand id #: {}".format(datasets, str(i)))
#     for name, value in zip(model.metrics_names, results):
#       print("%s: %.3f" % (name, value))
    
#     model_path = "saved_model/" + pth + "/my_model_txt_classifier_hub_with_three_percent_extra_data" + "_rand_" + str(i)
#     model.save(model_path)


# ## Train 10 models with 4% of new training data

# In[185]:



# gc.collect()

# class MyCustomCallback(tf.keras.callbacks.Callback):
#   def on_epoch_end(self, epoch, logs=None):
#     gc.collect()


# In[186]:


# i = 0
# datasets ="four_percent_additional_train_datasets"
# create_model_subdir(datasets)
# s = datasets.split("percent")
# pth = s[0] + "percent_models"

# for dataset in four_percent_additional_train_datasets:
#     i +=1 
    
#     train_examples_batch, train_labels_batch = next(iter(dataset.batch(10)))
    
#     embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
#     hub_layer = hub.KerasLayer(embedding, input_shape=[], 
#                                dtype=tf.string, trainable=True)
#     hub_layer(train_examples_batch[:3])

#     model = tf.keras.Sequential()
#     model.add(hub_layer)
#     model.add(tf.keras.layers.Dense(16, activation='relu'))
#     model.add(tf.keras.layers.Dense(1))

#     model.summary()
    
    
#     model.compile(optimizer='adam',
#               loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#               metrics=['accuracy'])
    
#     history = model.fit(dataset.shuffle(10000).batch(512),
#                     epochs=20,
#                     validation_data=validation_data.batch(512),
#                     verbose=0) #, use_multiprocessing=True)
    
#     results = model.evaluate(test_data.batch(512), verbose=0)

#     print("RESULTS for dataset - {} with rand id #: {}".format(datasets, str(i)))
#     for name, value in zip(model.metrics_names, results):
#       print("%s: %.3f" % (name, value))
    
#     model_path = "saved_model/" + pth + "/my_model_txt_classifier_hub_with_four_percent_extra_data" + "_rand_" + str(i)
#     model.save(model_path)


# ## Train 10 models with 5% of new training data

# In[187]:



# gc.collect()

# class MyCustomCallback(tf.keras.callbacks.Callback):
#   def on_epoch_end(self, epoch, logs=None):
#     gc.collect()


# In[188]:


# i = 0
# datasets ="five_percent_additional_train_datasets"
# create_model_subdir(datasets)
# s = datasets.split("percent")
# pth = s[0] + "percent_models"

# for dataset in five_percent_additional_train_datasets:
#     i +=1 
    
#     train_examples_batch, train_labels_batch = next(iter(dataset.batch(10)))
    
#     embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
#     hub_layer = hub.KerasLayer(embedding, input_shape=[], 
#                                dtype=tf.string, trainable=True)
#     hub_layer(train_examples_batch[:3])

#     model = tf.keras.Sequential()
#     model.add(hub_layer)
#     model.add(tf.keras.layers.Dense(16, activation='relu'))
#     model.add(tf.keras.layers.Dense(1))

#     model.summary()
    
    
#     model.compile(optimizer='adam',
#               loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#               metrics=['accuracy'])
    
#     history = model.fit(dataset.shuffle(10000).batch(512),
#                     epochs=20,
#                     validation_data=validation_data.batch(512),
#                     verbose=0) #, use_multiprocessing=True)
    
#     results = model.evaluate(test_data.batch(512), verbose=0)

#     print("RESULTS for dataset - {} with rand id #: {}".format(datasets, str(i)))
#     for name, value in zip(model.metrics_names, results):
#       print("%s: %.3f" % (name, value))
    
#     model_path = "saved_model/" + pth + "/my_model_txt_classifier_hub_with_five_percent_extra_data" + "_rand_" + str(i)
#     model.save(model_path)


# ## Train 10 models with 6% of new training data

# In[189]:



gc.collect()

class MyCustomCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    gc.collect()


# In[190]:


#TODO : 
# (1) DD this code to find the OOM instruction causing the kernel crash 
#(2) jupyter nbconvert to python and run for 6% to test. ... 


# In[191]:


# import tensorflow_text
# https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1
# https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim-with-oov/1 


# In[ ]:


i = 0
datasets ="six_percent_additional_train_datasets"
create_model_subdir(datasets)
s = datasets.split("percent")
pth = s[0] + "percent_models"

for dataset in six_percent_additional_train_datasets:
    i +=1 
    
    train_examples_batch, train_labels_batch = next(iter(dataset.batch(10)))
    
    embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
#     embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim-with-oov/1"
    hub_layer = hub.KerasLayer(embedding, input_shape=[], 
                               dtype=tf.string, trainable=True)
    hub_layer(train_examples_batch[:3])

    model = tf.keras.Sequential()
    model.add(hub_layer)
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(1))

    model.summary()
    #break
    
    
    model.compile(optimizer='adam',loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),metrics=['accuracy'])
    
    history = model.fit(dataset.shuffle(10000).batch(56),
                     epochs=20,
                     validation_data=validation_data.batch(56), #(512),
                     verbose=0) #, use_multiprocessing=True)
    
    results = model.evaluate(test_data.batch(56), verbose=0)

    print("RESULTS for dataset - {} with rand id #: {}".format(datasets, str(i)))
    for name, value in zip(model.metrics_names, results):
       print("%s: %.3f" % (name, value))
    
    model_path = "saved_model/" + pth + "/my_model_txt_classifier_hub_with_six_percent_extra_data" + "_rand_" + str(i)
    model.save(model_path)


# ## Train 10 models with 7% of new training data

# In[ ]:



# gc.collect()

# class MyCustomCallback(tf.keras.callbacks.Callback):
#   def on_epoch_end(self, epoch, logs=None):
#     gc.collect()


# In[ ]:


# i = 0
# datasets ="seven_percent_additional_train_datasets"
# create_model_subdir(datasets)
# s = datasets.split("percent")
# pth = s[0] + "percent_models"

# for dataset in seven_percent_additional_train_datasets:
#     i +=1 
    
#     train_examples_batch, train_labels_batch = next(iter(dataset.batch(10)))
    
#     embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
#     hub_layer = hub.KerasLayer(embedding, input_shape=[], 
#                                dtype=tf.string, trainable=True)
#     hub_layer(train_examples_batch[:3])

#     model = tf.keras.Sequential()
#     model.add(hub_layer)
#     model.add(tf.keras.layers.Dense(16, activation='relu'))
#     model.add(tf.keras.layers.Dense(1))

#     model.summary()
    
    
#     model.compile(optimizer='adam',
#               loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#               metrics=['accuracy'])
    
#     history = model.fit(dataset.shuffle(10000).batch(512),
#                     epochs=20,
#                     validation_data=validation_data.batch(512),
#                     verbose=0) #, use_multiprocessing=True)
    
#     results = model.evaluate(test_data.batch(512), verbose=0)

#     print("RESULTS for dataset - {} with rand id #: {}".format(datasets, str(i)))
#     for name, value in zip(model.metrics_names, results):
#       print("%s: %.3f" % (name, value))
    
#     model_path = "saved_model/" + pth + "/my_model_txt_classifier_hub_with_seven_percent_extra_data" + "_rand_" + str(i)
#     model.save(model_path)


# ## Train 10 models with 8% of new training data

# In[ ]:



# gc.collect()

# class MyCustomCallback(tf.keras.callbacks.Callback):
#   def on_epoch_end(self, epoch, logs=None):
#     gc.collect()


# In[ ]:


# i = 0
# datasets ="eight_percent_additional_train_datasets"
# create_model_subdir(datasets)
# s = datasets.split("percent")
# pth = s[0] + "percent_models"

# for dataset in eight_percent_additional_train_datasets:
#     i +=1 
    
#     train_examples_batch, train_labels_batch = next(iter(dataset.batch(10)))
    
#     embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
#     hub_layer = hub.KerasLayer(embedding, input_shape=[], 
#                                dtype=tf.string, trainable=True)
#     hub_layer(train_examples_batch[:3])

#     model = tf.keras.Sequential()
#     model.add(hub_layer)
#     model.add(tf.keras.layers.Dense(16, activation='relu'))
#     model.add(tf.keras.layers.Dense(1))

#     model.summary()
    
    
#     model.compile(optimizer='adam',
#               loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#               metrics=['accuracy'])
    
#     history = model.fit(dataset.shuffle(10000).batch(512),
#                     epochs=20,
#                     validation_data=validation_data.batch(512),
#                     verbose=0) #, use_multiprocessing=True)
    
#     results = model.evaluate(test_data.batch(512), verbose=0)

#     print("RESULTS for dataset - {} with rand id #: {}".format(datasets, str(i)))
#     for name, value in zip(model.metrics_names, results):
#       print("%s: %.3f" % (name, value))
    
#     model_path = "saved_model/" + pth + "/my_model_txt_classifier_hub_with_eight_percent_extra_data" + "_rand_" + str(i)
#     model.save(model_path)


# ## Train 10 models with 9% of new training data

# In[ ]:



# gc.collect()

# class MyCustomCallback(tf.keras.callbacks.Callback):
#   def on_epoch_end(self, epoch, logs=None):
#     gc.collect()


# In[ ]:


# i = 0
# datasets ="nine_percent_additional_train_datasets"
# create_model_subdir(datasets)
# s = datasets.split("percent")
# pth = s[0] + "percent_models"

# for dataset in nine_percent_additional_train_datasets:
#     i +=1 
    
#     train_examples_batch, train_labels_batch = next(iter(dataset.batch(10)))
    
#     embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
#     hub_layer = hub.KerasLayer(embedding, input_shape=[], 
#                                dtype=tf.string, trainable=True)
#     hub_layer(train_examples_batch[:3])

#     model = tf.keras.Sequential()
#     model.add(hub_layer)
#     model.add(tf.keras.layers.Dense(16, activation='relu'))
#     model.add(tf.keras.layers.Dense(1))

#     model.summary()
    
    
#     model.compile(optimizer='adam',
#               loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#               metrics=['accuracy'])
    
#     history = model.fit(dataset.shuffle(10000).batch(512),
#                     epochs=20,
#                     validation_data=validation_data.batch(512),
#                     verbose=0) #, use_multiprocessing=True)
    
#     results = model.evaluate(test_data.batch(512), verbose=0)

#     print("RESULTS for dataset - {} with rand id #: {}".format(datasets, str(i)))
#     for name, value in zip(model.metrics_names, results):
#       print("%s: %.3f" % (name, value))
    
#     model_path = "saved_model/" + pth + "/my_model_txt_classifier_hub_with_nine_percent_extra_data" + "_rand_" + str(i)
#     model.save(model_path)


# ## Train 10 models with 10% of new training data

# In[ ]:



# gc.collect()

# class MyCustomCallback(tf.keras.callbacks.Callback):
#   def on_epoch_end(self, epoch, logs=None):
#     gc.collect()


# In[ ]:


# i = 0
# datasets ="ten_percent_additional_train_datasets"
# create_model_subdir(datasets)
# s = datasets.split("percent")
# pth = s[0] + "percent_models"

# for dataset in ten_percent_additional_train_datasets:
#     i +=1 
    
#     train_examples_batch, train_labels_batch = next(iter(dataset.batch(10)))
    
#     embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
#     hub_layer = hub.KerasLayer(embedding, input_shape=[], 
#                                dtype=tf.string, trainable=True)
#     hub_layer(train_examples_batch[:3])

#     model = tf.keras.Sequential()
#     model.add(hub_layer)
#     model.add(tf.keras.layers.Dense(16, activation='relu'))
#     model.add(tf.keras.layers.Dense(1))

#     model.summary()
    
    
#     model.compile(optimizer='adam',
#               loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
#               metrics=['accuracy'])
    
#     history = model.fit(dataset.shuffle(10000).batch(512),
#                     epochs=20,
#                     validation_data=validation_data.batch(512),
#                     verbose=0) #, use_multiprocessing=True)
    
#     results = model.evaluate(test_data.batch(512), verbose=0)

#     print("RESULTS for dataset - {} with rand id #: {}".format(datasets, str(i)))
#     for name, value in zip(model.metrics_names, results):
#       print("%s: %.3f" % (name, value))
    
#     model_path = "saved_model/" + pth + "/my_model_txt_classifier_hub_with_ten_percent_extra_data" + "_rand_" + str(i)
#     model.save(model_path)


# In[ ]:





# In[ ]:





# In[ ]:




