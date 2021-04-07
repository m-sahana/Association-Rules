#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd


# In[2]:


from mlxtend.frequent_patterns import apriori,association_rules


# In[3]:


#with open as allows users to do line by line operations for single or multiple lines in the file
groceries = []
with open("D:\done\Association rules\groceries.csv") as f:
    groceries = f.read()


# In[5]:


groceries


# In[6]:


groceries = groceries.split("\n")
groceries


# In[7]:


groceries_list = []
for i in groceries:
    groceries_list.append(i.split(","))
groceries_list


# In[8]:


all_groceries_list = [i for item in groceries_list for i in item]
all_groceries_list


# In[9]:


from collections import Counter
#counts the number of times each item present in the datset
item_frequencies = Counter(all_groceries_list)
item_frequencies


# In[10]:


item_frequencies = sorted(item_frequencies.items(),key = lambda x:x[1])


# In[11]:


item_frequencies


# In[17]:


frequencies = list(([i[1] for i in item_frequencies]))
items = list(([i[0] for i in item_frequencies]))


# In[18]:


import matplotlib.pyplot as plt


# In[19]:


plt.figure(figsize=(8,6))
plt.scatter(items,frequencies)
plt.xlabel("items")
plt.ylabel("frequencies")


# In[20]:


groceries_series  = pd.DataFrame(pd.Series(groceries_list))


# In[21]:


groceries_series


# In[22]:


groceries_series = groceries_series.iloc[:9835,:] #to remove last empty element
groceries_series.columns = ["transactions"]


# In[23]:


groceries_series


# In[24]:


X = groceries_series['transactions'].str.join(sep='*').str.get_dummies(sep='*')
# creating a dummy columns for the each item in each transactions ... Using column names as item name


# In[25]:


X


# In[92]:


frequent_itemsets = apriori(X, min_support=0.005, max_len=2,use_colnames = True)
frequent_itemsets.head()


# In[86]:


frequent_itemsets = apriori(X, min_support=0.009, max_len=3,use_colnames = True)
frequent_itemsets.tail()


# In[87]:


frequent_itemsets = apriori(X, min_support=0.006, max_len=4,use_colnames = True)
frequent_itemsets.tail()


# In[88]:


frequent_itemsets = apriori(X, min_support=0.007, max_len=5,use_colnames = True)
frequent_itemsets.tail()


# In[89]:


frequent_itemsets = apriori(X, min_support=0.005, max_len=6,use_colnames = True)
frequent_itemsets.tail()


# In[31]:


frequent_itemsets = apriori(X, min_support=0.005, max_len=7,use_colnames = True)
frequent_itemsets.tail()


# In[32]:


frequent_itemsets = apriori(X, min_support=0.005, max_len=8,use_colnames = True)
frequent_itemsets.tail()


# In[38]:


frequent_itemsets.sort_values('support',ascending = False,inplace=True)


# In[33]:


association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)


# In[34]:


rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)
rules.sort_values('lift',ascending = False,inplace=True)


# In[35]:


rules.head(10)


# In[41]:


import matplotlib.pyplot as plt


# In[54]:


x=rules['support']
y=rules['confidence']
plt.scatter(x,y)


# In[75]:


x=rules['support']
y=rules['confidence']
plt.hist2d(x, y, cmap='Blues')


# In[76]:


#to eliminate redundancy
def to_list(i):
    return (sorted(list(i)))


# In[77]:


data = rules["antecedents"].apply(to_list)+rules["consequents"].apply(to_list)
data = data.apply(sorted)


# In[78]:


data


# In[79]:


rules_sets = list(data)


# In[80]:


rules_sets


# In[81]:


unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]
index_rules = []
for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))


# In[82]:


rules_no_redudancy  = rules.iloc[index_rules,:]


# In[83]:


rules_no_redudancy.sort_values('lift',ascending=False).head(10)


# In[ ]:




