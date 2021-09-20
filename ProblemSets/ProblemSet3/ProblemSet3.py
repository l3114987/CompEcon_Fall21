#!/usr/bin/env python
# coding: utf-8

# # ProblemSet3

# ##Importing data set

# In[3]:



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
mydata = pd.read_csv('table_1b.csv', header=0, index_col=1)
mydata1 = pd.read_csv('table_1b2.csv', header=0, index_col=0)
mydata.head(n=10)


# In[ ]:





# In[2]:


mydata1.head(n=10)


# ##Bar plot of Top 5% Patent Citation Inventor rate by State

# In[3]:


plt.style.use('ggplot')
fig1=mydata.plot(kind='bar', y="top5cit", use_index=True)
plt.ylabel('Top 5% Inventor rate')
plt.xlabel('State')
plt.title('Top 5% Inventor rate by State')
plt.savefig("top inventor.png")


# In[ ]:





# ##Colormap of Top 5% Patent Citation Inventor rate by State

# In[4]:


from urllib.request import urlopen
import json

with urlopen('https://eric.clst.org/assets/wiki/uploads/Stuff/gz_2010_us_040_00_500k.json') as response:
    state_map_data = json.load(response)
mydata.reset_index(inplace=True)
map = px.choropleth_mapbox(mydata, geojson=state_map_data,
                           locations="par_state",  # name of location in the dataframe
                           featureidkey="properties.NAME",   # name of the location in the geojson
                           color='top5cit',
                           color_continuous_scale="Jet",
                           mapbox_style="white-bg",
                           zoom=2, center = {"lat": 37.0902, "lon": -95.7129},
                           opacity=0.5,
                           title='Top 5% Most Cited Inventor Childhood state'
                          )
map.show()


# In[ ]:


map.write_image("colormap1.png")


# ###3D plot of Top 5% Most Cited Inventor Childhood by state and parental income quintile

# In[12]:


from mpl_toolkits.mplot3d import Axes3D
get_ipython().run_line_magic('matplotlib', 'notebook')
fig2 =plt.figure()
ax = fig2.add_subplot(111, projection = '3d')
ax.scatter(mydata1['par_stnum'], mydata1['pq'],
           mydata1['top5cit'], c='r', marker='o', alpha=0.1)
ax.view_init(elev=50., azim=15)
ax.set_xlabel('State')
ax.set_ylabel('Parental Income Quintile')
ax.set_zlabel('Top 5% Most Cited Inventor Rate')
plt.title("Top 5% Most Cited Inventor Rate and relation to State and Parental Income Quintile")
fig2.show()
fig2.savefig("3dplot.png")


# In[ ]:





# In[ ]:




