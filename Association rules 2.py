#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules


# In[4]:


movies_1=pd.read_csv("D:\projects\Association rules\my_movies.csv")


# In[5]:


movies_1.head()


# In[14]:


movies=movies_1.iloc[:,5:15]


# In[15]:


movies


# In[38]:


frequent_itemsets = apriori(movies, min_support=0.1, max_len=1,use_colnames = True)
frequent_itemsets


# In[39]:


frequent_itemsets = apriori(movies, min_support=0.2, max_len=2,use_colnames = True)
frequent_itemsets.tail()


# In[40]:


frequent_itemsets = apriori(movies, min_support=0.4, max_len=3,use_colnames = True)
frequent_itemsets.tail()


# In[41]:


frequent_itemsets = apriori(movies, min_support=0.6, max_len=4,use_colnames = True)
frequent_itemsets.tail()


# In[43]:


frequent_itemsets = apriori(movies, min_support=0.05, max_len=5,use_colnames = True)
frequent_itemsets.tail(1)


# In[31]:


frequent_itemsets.sort_values('support',ascending = False,inplace=True)


# In[32]:


association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)


# In[33]:


rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)
rules.sort_values('lift',ascending = False,inplace=True)


# In[44]:


rules


# In[35]:


import matplotlib.pyplot as plt


# In[36]:


x=rules['support']
y=rules['confidence']
plt.scatter(x,y)


# In[37]:


x=rules['support']
y=rules['confidence']
plt.hist2d(x, y, cmap='Blues')


# In[ ]:




