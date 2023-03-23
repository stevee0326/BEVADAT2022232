#!/usr/bin/env python
# coding: utf-8

# In[92]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[107]:


'''
FONTOS: Az első feladatáltal visszaadott DataFrame-et kell használni a további feladatokhoz. 
A függvényeken belül mindig készíts egy másolatot a bemenő df-ről, (new_df = df.copy() és ezzel dolgozz tovább.)
'''
# stats = {"country": ["Brazil", "Russia", "India", "China", "South Africa"],
#        "capital": ["Brasilia", "Moscow", "New Dehli", "Beijing", "Pretoria"],
#        "area": [8.516, 17.10, 3.286, 9.597, 1.221],
#        "population": [200.4, 143.5, 1252, 1357, 52.98] }
# df = dict_to_dataframe(stats)
#get_column(df,"country")
#get_top_two(df)
#population_density(df)
#plot_population(df)
#plot_area(df)


# In[95]:


'''
Készíts egy függvényt ami a bemeneti dictionary-ből egy DataFrame-et ad vissza.

Egy példa a bemenetre: test_dict
Egy példa a kimenetre: test_df
return type: pandas.core.frame.DataFrame
függvény neve: dict_to_dataframe
'''
def dict_to_dataframe(test_dict: dict):
    return pd.DataFrame(test_dict)


# In[94]:






# In[31]:


'''
Készíts egy függvényt ami a bemeneti DataFrame-ből vissza adja csak azt az oszlopot amelynek a neve a bemeneti string-el megegyező.

Egy példa a bemenetre: test_df, 'area'
Egy példa a kimenetre: test_df
return type: pandas.core.series.Series
függvény neve: get_column
'''
def get_column(df:pd.core.frame.DataFrame,colname: str):
    new_df = df.copy()
    a = new_df[colname]
    return a
    


# In[ ]:





# In[93]:


'''
Készíts egy függvényt ami a bemeneti DataFrame-ből vissza adja a két legnagyobb területű országhoz tartozó sorokat.

Egy példa a bemenetre: test_df
Egy példa a kimenetre: test_df
return type: pandas.core.frame.DataFrame
függvény neve: get_top_two
'''
def get_top_two(df : pd.core.frame.DataFrame):
    new_df = df.copy()
    return new_df.loc[np.argsort(new_df['area'])[:-3:-1]]
    


# In[ ]:





# In[46]:


'''
Készíts egy függvényt ami a bemeneti DataFrame-ből kiszámolja az országok népsűrűségét és eltárolja az eredményt egy új oszlopba ('density').
(density = population / area)

Egy példa a bemenetre: test_df
Egy példa a kimenetre: test_df
return type: pandas.core.frame.DataFrame
függvény neve: population_density
'''

def population_density(df:pd.core.frame.DataFrame):
    new_df = df.copy()
    density = new_df["population"] / new_df["area"]
    new_df["density"] = density
    return new_df


# In[ ]:





# In[100]:


'''
Készíts egy függvényt, ami a bemeneti Dataframe adatai alapján elkészít egy olyan oszlopdiagramot (bar plot),
ami vizualizálja az országok népességét.

Az oszlopdiagram címe legyen: 'Population of Countries'
Az x tengely címe legyen: 'Country'
Az y tengely címe legyen: 'Population (millions)'

Egy példa a bemenetre: test_df
Egy példa a kimenetre: fig
return type: matplotlib.figure.Figure
függvény neve: plot_population
'''
def plot_population(df: pd.core.frame.DataFrame):
    new_df = df.copy()
    fig, ax = plt.subplots()

    ax.bar(new_df["country"], new_df["population"])
    ax.set_xlabel("Country")
    ax.set_ylabel("Population")
    return fig
    
    


# In[ ]:





# In[105]:


'''
Készíts egy függvényt, ami a bemeneti Dataframe adatai alapján elkészít egy olyan kördiagramot,
ami vizualizálja az országok területét. Minden körcikknek legyen egy címe, ami az ország neve.

Az kördiagram címe legyen: 'Area of Countries'

Egy példa a bemenetre: test_df
Egy példa a kimenetre: fig
return type: matplotlib.figure.Figure
függvény neve: plot_area
'''

def plot_area(test_df):
    new_df = df.copy()
    fig, ax = plt.subplots()
    ax.set_title("Area of Countries")
    ax.pie(new_df["area"])
    return fig
    
    


# In[ ]:




