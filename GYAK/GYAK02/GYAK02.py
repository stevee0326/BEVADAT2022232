#!/usr/bin/env python
# coding: utf-8

# In[27]:


import numpy as np


# In[ ]:


create_array((2,2))
a = set_one(create_array((2,2)))
b = do_transpose(a)

a4 = np.array([0.1223, 0.1675])
round_array(a4)
a5 = np.array([[1, 0, 0], [1, 1, 1],[0, 0, 0]])
bool_array(a5)
a6 = np.array([[1, 0, 0], [1, 1, 1],[0, 0, 0]])
invert_bool_array(a6)
a7 = np.array([[1,2], [3,4]])
flatten(a7)


# In[29]:


#Készíts egy függvényt ami létre hoz egy nullákkal teli numpy array-t.
#Paraméterei: mérete (tuple-ként), default mérete pedig legyen egy (2,2)
#Be: (2,2)
#Ki: [[0,0],[0,0]]
#create_array()


# In[30]:


def create_array(tp : np.array) -> np.array:
    return np.zeros(tp)
    


# In[31]:


#Készíts egy függvényt ami a paraméterként kapott array-t főátlóját feltölti egyesekkel
#Be: [[1,2],[3,4]]
#Ki: [[1,2],[3,1]]
#set_one()


# In[32]:


def set_one(input_array : np.array)-> np.array:
    np.fill_diagonal(input_array,1)
    return input_array


# In[33]:


# Készíts egy függvényt ami transzponálja a paraméterül kapott mártix-ot:
# Be: [[1, 2], [3, 4]]
# Ki: [[1, 2], [3, 4]]
# do_transpose()


# In[34]:


def do_transpose(input_array: np.array)->np.array:
    return np.transpose(input_array)


# In[35]:


# Készíts egy olyan függvényt ami az array-ben lévő értékeket N tizenedjegyik kerekíti, ha nincs megadva ez a paraméter, akkor legyen az alapértelmezett a kettő 
# Be: [0.1223, 0.1675], 2
# Ki: [0.12, 0.17]
# round_array()


# In[49]:


def round_array(input_array : np.array) -> np.array:
    return input_array.round(2)
    


# In[36]:


# Készíts egy olyan függvényt, ami a bementként kapott 0 és 1 ből álló tömben a 0 - False-ra, az 1 True-ra cserélni
# Be: [[1, 0, 0], [1, 1, 1],[0, 0, 0]]
# Ki: [[ True False False], [ True  True  True], [False False False]]
# bool_array()


# In[55]:


def bool_array(input_array : np.array) -> np.array:
    return input_array.astype(dtype=bool)


# In[37]:


# Készíts egy olyan függvényt, ami a bementként kapott 0 és 1 ből álló tömben a 1 - False-ra az 0 True-ra cserélni
# Be: [[1, 0, 0], [1, 1, 1],[0, 0, 0]]
# Ki: [[ True False False], [ True  True  True], [False False False]]
# invert_bool_array()


# In[69]:


def invert_bool_array(input_array : np.array) -> np.array:
    return np.invert(input_array.astype(dtype=bool))


# In[38]:


# Készíts egy olyan függvényt ami a paraméterként kapott array-t kilapítja
# Be: [[1,2], [3,4]]
# Ki: [1,2,3,4]
# flatten()


# In[73]:


def flatten(input_array : np.array) -> np.array:
    return input_array.flatten()

