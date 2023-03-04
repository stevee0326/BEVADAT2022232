#!/usr/bin/env python
# coding: utf-8

# In[89]:


# Create a function that returns with a subsest of a list.
#The subset's starting and ending indexes should be set as input parameters (the list aswell).
#return type: list
#function name must be: subset
#input parameters: input_list,start_index,end_index

# input_list = [0,1,2,3,4,5,6,7,8,9]
# input_list2 = [[1,2],[3,4,5],[6,7,8,9]]
# input_listA = [0,1,2,3,4]
# input_listB = [5,6,7,8]
# input_TupleList = [(1,2),(3,4),(5,6,7)]
# matrixList = [[1,2,3],[1,2,7],[4,9,2],[6,0,5]]


# subset(input_list,2,7)
# every_nth(input_list,3)
# unique(input_list)
# flatten(input_list2)
# merge_lists(input_listA,input_listB)
# reverse_tuples(input_TupleList)
# remove_tuplicates(input_list)
# transpose(matrixList)
# split_into_chunks(input_list,3)

# dict1 = {
#     'a' : '1',
#     'b' : '2',
#     'c' : '3',
#     'd' : '4',
# }

# dict2 = {
#     'e' : '5',
#     'f' : '6',
#     'g' : '7',
#     'h' : '8',
# }

# dict3 = {
#     'l1' : [1,1,1,3,3,3],
#     'l2' : [2,2,2,4,4,4]
# }

# merge_dicts(dict1,dict2)
# mean_key_value(dict3)


# In[6]:


def subset(input_list,start_index,end_index):
    return input_list[start_index:end_index]


# In[ ]:


#Create a function that returns every nth element of a list.
#return type: list
#function name must be: every_nth
#input parameters: input_list,step_size


# In[9]:


def every_nth(input_list,step_size):
    return input_list[::step_size]


# In[17]:


#Create a function that can decide whether a list contains unique values or not
#return type: bool
#function name must be: unique
#input parameters: input_list


# In[20]:


def unique(input_list):
    set1 = set(input_list)
    if len(set1) == len(input_list):
        return True
    else:
        return False
    


# In[ ]:


#Create a function that can flatten a nested list ([[..],[..],..])
#return type: list
#fucntion name must be: flatten
#input parameters: input_list


# In[23]:


def flatten(input_list):
    output_list = []
    for x in input_list:
        for y in x:
            output_list.append(y)
    return output_list


# In[ ]:


#Create a function that concatenates n lists
#return type: list
#function name must be: merge_lists
#input parameters: *args


# In[27]:


def merge_lists(*args):
    output_list = []
    for arg in args:
        for x in arg:
            output_list.append(x)
    return output_list


# In[ ]:


#Create a function that can reverse a list of tuples
#example [(1,2),...] => [(2,1),...]
#return type: list
#fucntion name must be: reverse_tuples
#input parameters: input_list


# In[35]:


def reverse_tuples(input_list):
    output_list = []
    for x in input_list:
        output_list.append(x[::-1])
    return output_list


# In[ ]:


#Create a function that removes duplicates from a list
#return type: list
#fucntion name must be: remove_tuplicates
#input parameters: input_list


# In[37]:


def remove_tuplicates(input_list):
    return set(input_list)


# In[ ]:


#Create a function that transposes a nested list (matrix)
#return type: list
#function name must be: transpose
#input parameters: input_list


# In[47]:


def transpose(input_list):
    matrix1 = []
    for x in range(len(input_list[0])):
        l = []
        for y in range(len(input_list)):
            l.append(input_list[y][x])
        matrix1.append(l)
    return matrix1
    


# In[ ]:


#Create a function that can split a nested list into chunks
#chunk size is given by parameter
#return type: list
#function name must be: split_into_chunks
#input parameters: input_list,chunk_size


# In[71]:


def split_into_chunks(input_list,chunk_size):
    output_list = []
    for i in range(0, len(input_list), chunk_size):
        output_list.append(input_list[i:i+chunk_size])
    return output_list


# In[ ]:


#Create a function that can merge n dictionaries
#return type: dictionary
#function name must be: merge_dicts
#input parameters: *dict


# In[80]:


def merge_dicts(*dict):
    output_dicts = {}
    for d in dict:
        for k, v in d.iteritems():  # d.items() in Python 3+
            output_dicts.setdefault(k, []).append(v)
    return output_dicts


# In[ ]:


#Create a function that receives a list of integers and sort them by parity
#and returns with a dictionary like this: {"even":[...],"odd":[...]}
#return type: dict
#function name must be: by_parity
#input parameters: input_list


# In[ ]:





# In[ ]:


#Create a function that receives a dictionary like this: {"some_key":[1,2,3,4],"another_key":[1,2,3,4],....}
#and return a dictionary like this : {"some_key":mean_of_values,"another_key":mean_of_values,....}
#in short calculates the mean of the values key wise
#return type: dict
#function name must be: mean_key_value
#input parameters: input_dict


# In[88]:


def mean_key_value(input_dict):
    output_dicts = {}
    for k, v in input_dict.iteritems():
        s = 0
        for x in v:
            s = s + x
        avg = s / len(v)
        output_dicts.setdefault(k, avg)
    return output_dicts
        


# In[ ]:


#If all the functions are created convert this notebook into a .py file and push to your repo

