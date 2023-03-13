#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np


# In[134]:


#FONTOS!!!

# CSAK OTT LEHET HASZNÁLNI FOR LOOP-OT AHOL A FELADAT KÜLÖN KÉRI!
# [1,2,3,4] --> ezek az értékek np.array-ek. Ahol listát kérek paraméterként ott külön ki fogom emelni!
# Ha végeztél a feladatokkal, akkor notebook-ot alakítsd át .py.
# A FÁJLBAN CSAK A FÜGGVÉNYEK LEGYENEK! (KOMMENTEK MARADHATNAK)
# a1 = np.array([[1,2],[3,4]])
#a2 = np.array([7,8,9],[9,8,7])
#column_swap(a1)

# b1 = np.array([7, 8, 9])
# b2 = np.array([9, 8, 7])
#compare_two_array(b1, b2)

# c1 = np.array([[[1,2,3], [4,5,6]]])

#get_array_shape(c1)

# d1 = np.array([1, 1, 1, 1])
# encode_Y(d1, 4)
# decode_Y(np.array([[0,1,0,0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]))

# eval_classification(np.array(['alma', 'körte', 'szilva']),np.array([0.2, 0.2, 0.6]))
# repalce_odd_numbers(np.array([1,2,3,4,5,6]))

# replace_by_value(np.array([1, 2, 5, 0]), 1)
# array_multi(np.array([1,2,3,4]))
# array_multi_2d(np.array([[1, 2], [3, 4]]))
# add_border(np.array([[1,2,2],[3,4,2],[3,4,2]]))

# list_days('2023-03','2023-04')
#curr_date()
#sec_from_1970()


# In[ ]:


# Írj egy olyan fügvényt, ami megfordítja egy 2d array oszlopait. Bemenetként egy array-t vár.
# Be: [[1,2],[3,4]]
# Ki: [[2,1],[4,3]]
# column_swap()
def column_swap(input_array : np.array) -> np.array:
    return np.flip(input_array, axis=1)
    


# In[ ]:


# Készíts egy olyan függvényt ami összehasonlít két array-t és adjon vissza egy array-ben, hogy hol egyenlőek 
# Pl Be: [7,8,9], [9,8,7] 
# Ki: [1]
# compare_two_array()
# egyenlő elemszámúakra kell csak hogy működjön
def compare_two_array(input_array1 : np.array,input_array2 : np.array):
    return np.where(input_array1 == input_array2)[0]


# In[32]:


# Készíts egy olyan függvényt, ami vissza adja string-ként a megadott array dimenzióit:
# Be: [[1,2,3], [4,5,6]]
# Ki: "sor: 2, oszlop: 3, melyseg: 1"
# get_array_shape()
# 3D-vel még műküdnie kell!, 

def get_array_shape(nput_array1 : np.array):
    s = nput_array1.shape
    th = 1 if len(s) < 3 else s[0]
    str1 = "sor: " + str(s[-2]) + ", oszlop: " + str(s[-1]) + ", melyseg: " + str(th)
    return str1


# In[59]:


# Készíts egy olyan függvényt, aminek segítségével elő tudod állítani egy neurális hálózat tanításához szükséges pred-et egy numpy array-ből. 
# Bementként add meg az array-t, illetve hogy mennyi class-od van. Kimenetként pedig adjon vissza egy 2d array-t, ahol a sorok az egyes elemek. Minden nullákkal teli legyen és csak ott álljon egyes, ahol a bementi tömb megjelöli. 
# Pl. ha 1 van a bemeneten és 4 classod van, akkor az adott sorban az array-ban a [1] helyen álljon egy 1-es, a többi helyen pedig 0.
# Be: [1, 2, 0, 3], 4
# Ki: [[0,1,0,0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]
# encode_Y()

def encode_Y(input_array1 : np.array, size):
    results = np.zeros((len(input_array1), size))
    for i, a in enumerate(input_array1):
        results[i, a] = 1
    return results


# In[48]:


# A fenti feladatnak valósítsd meg a kiértékelését. Adj meg a 2d array-t és adj vissza a decodolt változatát
# Be:  [[0,1,0,0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]
# Ki:  [1, 2, 0, 3]
# decode_Y()

def decode_Y(input_array1: np.array) -> np.array:
    results = np.zeros(len(input_array1))
    for i, a in enumerate(input_array1):
        results[i] = np.where(a == 1)[0][0]
    return results


# In[64]:


# Készíts egy olyan függvényt, ami képes kiértékelni egy neurális háló eredményét! Bemenetként egy listát és egy array-t és adja vissza azt az elemet, aminek a legnagyobb a valószínüsége(értéke) a listából.
# Be: ['alma', 'körte', 'szilva'], [0.2, 0.2, 0.6]. # Az ['alma', 'körte', 'szilva'] egy lista!
# Ki: 'szilva'
# eval_classification()

def eval_classification(classes: np.array,probs: np.array):
    return classes[probs.argmax(axis=0)]


# In[72]:


# Készíts egy olyan függvényt, ahol az 1D array-ben a páratlan számokat -1-re cseréli
# Be: [1,2,3,4,5,6]
# Ki: [-1,2,-1,4,-1,6]
# repalce_odd_numbers()

def repalce_odd_numbers(input_array1: np.array):
    return np.where(input_array1 % 2, -1, input_array1)


# In[74]:


# Készíts egy olyan függvényt, ami egy array értékeit -1 és 1-re változtatja, attól függően, hogy az adott elem nagyobb vagy kisebb a paraméterként megadott számnál.
# Ha a szám kisebb mint a megadott érték, akkor -1, ha nagyobb vagy egyenlő, akkor pedig 1.
# Be: [1, 2, 5, 0], 2
# Ki: [-1, 1, 1, -1]
# replace_by_value()
def replace_by_value(input_array1: np.array, f):
    return np.where(input_array1 < f, -1, 1)
    


# In[79]:


# Készíts egy olyan függvényt, ami egy array értékeit összeszorozza és az eredményt visszaadja
# Be: [1,2,3,4]
# Ki: 24
# array_multi()
# Ha több dimenziós a tömb, akkor az egész tömb elemeinek szorzatával térjen vissza
def array_multi(input_array1: np.array):
    return np.prod(input_array1)


# In[81]:


# Készíts egy olyan függvényt, ami egy 2D array értékeit összeszorozza és egy olyan array-el tér vissza, aminek az elemei a soroknak a szorzata
# Be: [[1, 2], [3, 4]]
# Ki: [2, 12]
# array_multi_2d()
def array_multi_2d(input_array1: np.array):
    return np.prod(input_array1, axis=1)
    


# In[85]:


# Készíts egy olyan függvényt, amit egy meglévő numpy array-hez készít egy bordert nullásokkal. Bementként egy array-t várjon és kimenetként egy array jelenjen meg aminek van border-je
# Be: [[1,2],[3,4]]
# Ki: [[0,0,0,0],[0,1,2,0],[0,3,4,0],[0,0,0,0]]
# add_border()
def add_border(input_array1: np.array):
    i, k = input_array1.shape
    result = np.zeros((i+2, k+2))
    result[1:-1, 1:-1] = input_array1
    return result
    


# In[ ]:


# A KÖTVETKEZŐ FELADATOKHOZ NÉZZÉTEK MEG A NUMPY DATA TYPE-JÁT!


# In[133]:


# Készíts egy olyan függvényt ami két dátum között felsorolja az összes napot és ezt adja vissza egy numpy array-ben. A fgv ként str vár paraméterként 'YYYY-MM' formában.
# Be: '2023-03', '2023-04'  # mind a kettő paraméter str.
# Ki: ['2023-03-01', '2023-03-02', .. , '2023-03-31',]
# list_days()

def list_days(date1, date2):
    return np.arange(date1, date2, dtype='datetime64[D]')


# In[130]:


# Írj egy fügvényt ami vissza adja az aktuális dátumot az alábbi formában: YYYY-MM-DD. Térjen vissza egy 'numpy.datetime64' típussal.
# Be:
# Ki: 2017-03-24

#fuggveny neve??
def curr_date():
    return np.datetime64('today','D')


# In[122]:


# Írj egy olyan függvényt ami visszadja, hogy mennyi másodperc telt el 1970 január 01. 00:02:00 óta. Int-el térjen vissza
# Be: 
# Ki: másodpercben az idó, int-é kasztolva
# sec_from_1970()

def sec_from_1970():
    return int((np.datetime64('now') - np.datetime64("1970-01-01T00:02:00"))/np.timedelta64(1,'s'))

