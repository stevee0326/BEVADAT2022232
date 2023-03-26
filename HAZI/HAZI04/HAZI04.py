#!/usr/bin/env python
# coding: utf-8

# In[151]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random


# In[217]:


'''
FONTOS: Az első feladatáltal visszaadott DataFrame-et kell használni a további feladatokhoz. 
A függvényeken belül mindig készíts egy másolatot a bemenő df-ről, (new_df = df.copy() és ezzel dolgozz tovább.)
'''
#df = csv_to_df("StudentsPerformance")
#df = pd.read_csv('StudentsPerformance.csv')
#capitalize_columns(df)
#math_passed_count(df)
#did_pre_course(df)
#average_scores(df)
#pd.read_csv('StudentsPerformance.csv')
#add_age(df)
#female_top_score(df)
#add_grade(df)


# In[ ]:


'''
Készíts egy függvényt, ami egy string útvonalat vár paraméterként, és egy DataFrame ad visszatérési értékként.

Egy példa a bemenetre: 'test_data.csv'
Egy példa a kimenetre: df_data
return type: pandas.core.frame.DataFrame
függvény neve: csv_to_df
'''


# In[2]:


def csv_to_df(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


# In[ ]:


'''
Készíts egy függvényt, ami egy DataFrame-et vár paraméterként, 
és átalakítja azoknak az oszlopoknak a nevét nagybetűsre amelyiknek neve nem tartalmaz 'e' betüt.

Egy példa a bemenetre: df_data
Egy példa a kimenetre: df_data_capitalized
return type: pandas.core.frame.DataFrame
függvény neve: capitalize_columns
'''


# In[89]:


def capitalize_columns(df: pd.DataFrame) -> pd.core.frame.DataFrame:
    new_df = df.copy()
    cols = []
    for x in new_df.columns:
        if('e' in x):
            cols.append(x)
        else:
            cols.append(x.upper())
    new_df.columns = cols
    return new_df    


# In[ ]:


'''
Készíts egy függvényt, ahol egy szám formájában vissza adjuk, hogy hány darab diáknak sikerült teljesíteni a matek vizsgát.
(legyen az átmenő ponthatár 50).

Egy példa a bemenetre: df_data
Egy példa a kimenetre: 5
return type: int
függvény neve: math_passed_count
'''


# In[102]:


def math_passed_count(df: pd.DataFrame) -> int:
    new_df = df.copy()
    return np.count_nonzero(new_df['math score'] >= 50)


# In[132]:


'''
Készíts egy függvényt, ahol Dataframe ként vissza adjuk azoknak a diákoknak az adatait (sorokat), akik végeztek előzetes gyakorló kurzust.

Egy példa a bemenetre: df_data
Egy példa a kimenetre: df_did_pre_course
return type: pandas.core.frame.DataFrame
függvény neve: did_pre_course
'''
def did_pre_course(df: pd.DataFrame) -> pd.DataFrame:
    new_df = df.copy()
    return new_df[new_df['test preparation course'] == 'completed']


# In[ ]:





# In[146]:


'''
Készíts egy függvényt, ahol a bemeneti Dataframet a diákok szülei végzettségi szintjei alapján csoportosításra kerül,
majd aggregációként vegyük, hogy átlagosan milyen pontszámot értek el a diákok a vizsgákon.

Egy példa a bemenetre: df_data
Egy példa a kimenetre: df_average_scores
return type: pandas.core.frame.DataFrame
függvény neve: average_scores
'''
def average_scores(df: pd.DataFrame) -> pd.DataFrame:
    new_df = df.copy()
    return new_df.groupby("parental level of education").mean()


# In[ ]:





# In[164]:


'''
Készíts egy függvényt, ami a bementeti Dataframet kiegészíti egy 'age' oszloppal, töltsük fel random 18-66 év közötti értékekkel.
A random.randint() függvényt használd, a random sorsolás legyen seedleve, ennek értéke legyen 42.

Egy példa a bemenetre: df_data
Egy példa a kimenetre: df_data_with_age
return type: pandas.core.frame.DataFrame
függvény neve: add_age
'''
def add_age(df: pd.DataFrame) -> pd.DataFrame:
    new_df = df.copy()
    random.seed(42)
    new_df['age'] = [random.randint(18, 67) for i in range(new_df.shape[0])]
    return new_df


# In[ ]:





# In[206]:


'''
Készíts egy függvényt, ami vissza adja a legjobb teljesítményt elérő női diák pontszámait.

Egy példa a bemenetre: df_data
Egy példa a kimenetre: (99,99,99) #math score, reading score, writing score
return type: tuple
függvény neve: female_top_score
'''
def female_top_score(df: pd.DataFrame) -> tuple:
    new_df = df.copy()
    new_df = new_df[new_df['gender'] == 'female']
    best = new_df[["math score", "reading score", "writing score"]].max()
    return tuple(best)


# In[ ]:





# In[216]:


'''
Készíts egy függvényt, ami a bementeti Dataframet kiegészíti egy 'grade' oszloppal. 
Számoljuk ki hogy a diákok hány százalékot ((math+reading+writing)/300) értek el a vizsgán, és osztályozzuk őket az alábbi szempontok szerint:

90-100%: A
80-90%: B
70-80%: C
60-70%: D
<60%: F

Egy példa a bemenetre: df_data
Egy példa a kimenetre: df_data_with_grade
return type: pandas.core.frame.DataFrame
függvény neve: add_grade
'''
def add_grade(df: pd.DataFrame) -> pd.DataFrame :
    new_df = df.copy()
    percentages = (new_df['math score'] + new_df['reading score'] + new_df['writing score']) / 3
    new_df['grade'] = pd.cut(percentages,bins=[0, 60, 70, 80, 90, 100],labels=["F", "D", "C", "B", "A"])
    return new_df


# In[ ]:





# In[ ]:


'''
Készíts egy függvényt, ami a bemeneti Dataframe adatai alapján elkészít egy olyan oszlop diagrammot,
ami vizualizálja a nemek által elért átlagos matek pontszámot.

Oszlopdiagram címe legyen: 'Average Math Score by Gender'
Az x tengely címe legyen: 'Gender'
Az y tengely címe legyen: 'Math Score'

Egy példa a bemenetre: df_data
Egy példa a kimenetre: fig
return type: matplotlib.figure.Figure
függvény neve: math_bar_plot
'''


# In[239]:


def math_bar_plot(df: pd.DataFrame) -> plt.Figure:
    new_df = df.copy()
    dfGroupedbyMathAvg = new_df.groupby('gender')['math score'].mean()
    fig, ax = plt.subplots()
    ax.bar(dfGroupedbyMathAvg.index, dfGroupedbyMathAvg.values)
    ax.set_title("Average Math Score by Gender")
    ax.set_xlabel('Gender')
    ax.set_ylabel('Math Score')
    return fig


# In[ ]:


''' 
Készíts egy függvényt, ami a bemeneti Dataframe adatai alapján elkészít egy olyan histogramot,
ami vizualizálja az elért írásbeli pontszámokat.

A histogram címe legyen: 'Distribution of Writing Scores'
Az x tengely címe legyen: 'Writing Score'
Az y tengely címe legyen: 'Number of Students'

Egy példa a bemenetre: df_data
Egy példa a kimenetre: fig
return type: matplotlib.figure.Figure
függvény neve: writing_hist
'''


# In[246]:


def writing_hist(df: pd.DataFrame) -> plt.Figure:
    new_df = df.copy()
    fig, ax = plt.subplots()
    ax.set_title('Distribution of Writing Scores')
    ax.set_xlabel('Writing Score')
    ax.set_ylabel('Number of Students')
    ax.hist(new_df['writing score'])
    return fig


# In[251]:


''' 
Készíts egy függvényt, ami a bemeneti Dataframe adatai alapján elkészít egy olyan kördiagramot,
ami vizualizálja a diákok etnikum csoportok szerinti eloszlását százalékosan.

Érdemes megszámolni a diákok számát, etnikum csoportonként,majd a százalékos kirajzolást az autopct='%1.1f%%' paraméterrel megadható.
Mindegyik kör szelethez tartozzon egy címke, ami a csoport nevét tartalmazza.
A diagram címe legyen: 'Proportion of Students by Race/Ethnicity'

Egy példa a bemenetre: df_data
Egy példa a kimenetre: fig
return type: matplotlib.figure.Figure
függvény neve: ethnicity_pie_chart
'''

def ethnicity_pie_chart(df: pd.DataFrame) -> plt.Figure:
    new_df = df.copy()
    fig, ax = plt.subplots()
    ax.set_title("Proportion of Students by Race/Ethnicity")
    grouped = new_df.groupby("race/ethnicity")["race/ethnicity"].count()
    labels = grouped.index
    values = grouped.values
    ax.pie(values, labels = labels, autopct='%1.1f%%')
    return fig


# In[ ]:




