#!/usr/bin/env python
# coding: utf-8

# Consider a model of a one-to-one matching market representing radio station mergers. Each year there
# is a national market where radio station owners target new stations. These markets are independent across
# years.
# There are two models to estimate for the current assignment. First, I estimate the parameters of a payoff function without transfers that represent the relative importance of corporate ownership and geographic proximity compared to size sorting.
# The payoff to the merger between radio station buyer b and target t in market m is given by:
# 
# $ f_m(b,t) = x_{1bm}y_{1tm}+\alpha x_{2bm}y_{1tm}+\beta distance_{btm}+\epsilon_{btm}$
# 
# Then, I estimate the version of the above model with target characteristics and transfers (the prices pay to aquire the target station). I use the data on the of the merger and a different inequality in my score function.
# 
# $ f_m(b,t) = \delta x_{1bm}y_{1tm}+\alpha x_{2bm}y_{1tm}+\gamma HHI_{tm}+\beta distance_{btm}+\epsilon_{btm}$
# 
# The ultimate goal of the current assugment is to estimate the parameters of the models, ($\alpha, \beta$), and ($\delta, \alpha, \gamma, \beta$) and the maximum score estimators of each model.

# In[ ]:


#Importing packages
import numpy as np
import pandas as pd
import geopy as distance
from geopy.distance import geodesic
import scipy.optimize as opt
from scipy.optimize import differential_evolution


# In[ ]:


#Loading the data file 
df = pd.read_csv('radio_merger_data.csv')
df.head(10)
df.describe()


# In[ ]:


#Putting the price and population variables into miilons of dollars and people
df['price_mil']=df['price']/1000000
df['pop_mil']=df['population_target']/1000000


# In[ ]:


#The first model
def myfun1(parameters,*data):
    '''
    Args:
    parameters = a tuple including 2 parameters to estimate
    data = data set in use
    
    Returns:
    mse_obj_fun: Nagtive value of the objective function for MSE
    
    '''
    alpha, beta = parameters
    
    # Restructuring the data used into the f(b,t) matrices while spliting the data into two groups (by year)
   
    
    data07 = data[0]
    f07 = np.zeros((data07.shape[0],data07.shape[0]))
    
    data08 = data[1]
    f08 = np.zeros((data08.shape[0],data08.shape[0]))
   
    # Each of the loops created in this funtion is used to actually sum the value of a merger for each buyer-target pair. 
    # (i,j) in either matrix corresponds to the value of the merger between i and j in a given year market (2007, 2008)

    for i in range(data07.shape[0]):
        for j in range(data07.shape[0]):
            distance =geodesic((data07.loc[i,'buyer_lat'],data07.loc[i,'buyer_long']),
                                (data07.loc[j,'target_lat'],data07.loc[j,'target_long'])).miles
            f07[i,j]=((data07.loc[i,'num_stations_buyer']*data07.loc[j,'pop_mil'])
                  +(alpha*data07.loc[i,'corp_owner_buyer']*data07.loc[j,'pop_mil'])
                  +(beta*distance))
    for i in range(data08.shape[0]):
        for j in range(data08.shape[0]):
            distance =geodesic((data08.iloc[0,3],data08.iloc[0,4]),
                                (data08.iloc[0,5],data08.iloc[0,6])).miles
            f08[i,j]=((data08.iloc[i,9]*data08.iloc[j,13])
                  +(alpha*data08.iloc[i,11]*data08.iloc[j,13])
                  +(beta*beta*distance))
             
  
    #Now Compute the objective function consisting of the parameters (alpha and beta) by summing the indicator fucntions of each year
    
    obj07 =0
    for i in range(data07.shape[0]-1):
        for j in range(i+1, data07.shape[0]):
            actual=f07[i,i]+f07[j,j] # actual merger value
            counterfactual=f07[i,j]+f07[j,i] #counterfactual merger value
            if (actual >=counterfactual):
                obj07 +=1
   
    
    obj08 =0
    for i in range(data08.shape[0]-1):
        for j in range(i+1, data08.shape[0]):
            actual=f08[i,i]+f08[j,j] # actual merger value
            counterfactual=f08[i,j]+f08[j,i] #counterfactual merger value
            if (actual >=counterfactual):
                obj08 +=1
                
    mse_obj_fun = obj07 + obj08
    return -mse_obj_fun


# In[ ]:


#Spliting the main data given the independence between the two years (2007, 2008)
df07 =df[df['year']==2007]
df08 =df[df['year']==2008]


# In[ ]:


# Run a optimization rutine with myfun1 I made above
bounds = [(-10000, 10000), (-10000, 10000)]
args = (df07, df08)
obj_min = 1e+100

# I run the optimization routine five times (it took 57mins )
for i in range(5):
    max_score = differential_evolution(myfun1, bounds, args = args, tol = 1e-15)
    if(max_score['fun'] < obj_min):
        obj_min = max_score['fun']
        alpha_min = max_score['x'][0]
        beta_min = max_score['x'][1]

print('Estimates for the first model without price')

print("Minimum function value = ", obj_min)
print("Optimum alpha = ", alpha_min)
print("Optimum beta = ", beta_min) 


# In[ ]:


#The second model including price
def myfun2(parameters,*data):
    '''
    Args:
    parameters = a tuple including 4 parameters to estimate
    data = data set in use
    
    Returns:
    mse_obj_fun: Nagtive value of the objective function for MSE
    
    '''
     
    delta, alpha, gamma, beta = parameters
    
   # Restructuring the data used into the f(b,t) matrices while spliting the data into two groups (by year)
    
    data07 = data[0]
    f07 = np.zeros((data07.shape[0],data07.shape[0]))
    
    data08 = data[1]
    f08 = np.zeros((data08.shape[0],data08.shape[0]))
    
    # Each of the loops created in this funtion is used to actually sum the value of a merger for each buyer-target pair. 
    # (i,j) in either matrix corresponds to the value of the merger between i and j in a given year market (2007, 2008)
    
    for i in range(data07.shape[0]):
        for j in range(data07.shape[0]):
            distance = geodesic((data07.loc[i,'buyer_lat'],data07.loc[i,'buyer_long']),
                                (data07.loc[j,'target_lat'],data07.loc[j,'target_long'])).miles
            f07[i,j] = ((delta * data07.loc[i,'num_stations_buyer'] * data07.loc[j,'pop_mil'])
                         + (alpha * data07.loc[i,'corp_owner_buyer'] * 
                            data07.loc[j,'pop_mil']) + 
                         (gamma * data07.loc[j,'hhi_target']) + (beta * distance))
    
    for i in range(data08.shape[0]):
        for j in range(data08.shape[0]):
            distance = geodesic((data08.iloc[0,3], data08.iloc[0,4]),
                                (data08.iloc[0,5], data08.iloc[0,6])).miles
            f08[i,j] = ((delta * data08.iloc[i,9] * data08.iloc[j,13])
                         + (alpha * data08.iloc[i,11] * 
                            data08.iloc[j,13]) + (gamma * data08.iloc[j,8]) + 
                         (beta * distance))

    
    #Now Compute the objective function consisting of the parameters (delta,alpha, gamma and beta)by summing the indicator fucntions of each year
    
    obj07 = 0
    for i in range(f07.shape[0] - 1):
        for j in range(i + 1, f07.shape[0]):
            LHS_1 = f07[i,i] - f07[i,j]  
            RHS_1 = data07.loc[i,'price_mil'] - data07.loc[j,'price_mil'] 
            LHS_2 = f07[j,j] - f07[j,i]
            RHS_2 = data07.loc[j,'price_mil'] - data07.loc[i,'price_mil']
            if((LHS_1 >= RHS_1) & (LHS_2 >= RHS_2)):    
                obj07 += 1
    
    obj08 = 0
    for i in range(f08.shape[0] - 1):
        for j in range(i + 1, f08.shape[0]):
            LHS_1 = f08[i,i] - f08[i,j]
            RHS_1 = data08.iloc[i,12] - data08.iloc[j,12]
            LHS_2 = f08[j,j] - f08[j,i]
            RHS_2 = data08.iloc[j,12] - data08.iloc[i,12]
            if((LHS_1 >= RHS_1) & (LHS_2 >= RHS_2)):
                obj08 += 1
                
    mse_obj_fun = obj07 + obj08
    return -mse_obj_fun


# In[ ]:


# Run a optimization rutine with myfun2 I made above
bounds = [(-10000, 10000), (-10000, 10000), (-10000, 10000), (-10000, 10000)] # Using the same bounds used for the first model
args = (df07, df08)

# I run the optimization routine five times (it took 9 hours 20 mins )
for i in range(5):
    max_score2 = differential_evolution(myfun2, bounds, args = args, tol = 1e-15)
    print('Estimates for the second model with price')
    print("Minimum function value = ", max_score2['fun'])
    print("Optimum delta = ", max_score2['x'][0])
    print("Optimum alpha = ", max_score2['x'][1])
    print("Optimum gamma = ", max_score2['x'][2]) 
    print("Optimum beta = ", max_score2['x'][3]) 


# In[ ]:




