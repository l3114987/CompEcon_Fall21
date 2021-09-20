
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

####Importing data set####
mydata = pd.read_csv('table_1b.csv', header=0, index_col=1)
mydata1 = pd.read_csv('table_1b2.csv', header=0, index_col=0)
mydata.head(n=10)
mydata1.head(n=10)

plt.style.use('ggplot')
fig1=mydata.plot(kind='bar', y="top5cit", use_index=True)
plt.ylabel('Top 5% Inventor rate')
plt.xlabel('State')
plt.title('Top 5% Inventor rate by State')
fig1.write_image("top inventor.png")
