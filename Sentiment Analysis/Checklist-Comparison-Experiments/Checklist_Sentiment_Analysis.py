#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pickle


# In[17]:


with open('religion_tests.pkl', 'rb') as handle:
    religion_tests = pickle.load(handle)
    
with open('nationality_tests.pkl', 'rb') as handle:
    nationality_tests = pickle.load(handle)
    
with open('race_tests.pkl', 'rb') as handle:
    race_tests = pickle.load(handle)
    
with open('sexuality_tests.pkl', 'rb') as handle:
    sexuality_tests = pickle.load(handle)


# In[18]:


print(len(religion_tests[0]))
print(len(nationality_tests[0]))
print(len(race_tests[0]))
print(len(sexuality_tests[0]))


# In[ ]:





# In[ ]:




