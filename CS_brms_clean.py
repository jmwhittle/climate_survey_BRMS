#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 08:42:33 2019

@author: jasonwhittle
"""

import numpy as np
import pandas as pd

data = pd.read_csv('cs2019_data.csv')
data_key = data[0:2]

data = data.drop([0,1], axis = 0)
data = data.iloc[:,17:]

data.head()
data.info()


# combining campus statuses into one variable
data['temp'] = data["Q3"].combine_first(data["Q4"])
data['status'] = data["temp"].combine_first(data["Q5"])


# turning Q9 into three distinct outcome dummies
Q9 = data["Q9"].str.split(',', expand = True) # this doesn't split them clean shouldn't use get_dummies

# Custome conditional dummies for Q9
# welcome
def welcome(x, y, z):
    if x == 'I feel welcome at SLCC.':
        return 1
    elif y == 'I feel welcome at SLCC.':
        return 1
    elif z == 'I feel welcome at SLCC.':
        return 1
    else:
        return 0

func_wl = np.vectorize(welcome)
welcome = func_wl(Q9[0], Q9[1], Q9[2])
Q9['welcome'] = welcome
# code for welcome is probably not needed but was used for consistency 
# np.where(Q9[0] == 'I feel welcome at SLCC.', 1, 0) will do this just for Welcome just as well. 

# belong
def belong(x, y, z):
    if x == 'I feel that I belong at SLCC.':
        return 1
    elif y == 'I feel that I belong at SLCC.':
        return 1
    elif z == 'I feel that I belong at SLCC.':
        return 1
    else:
        return 0

func_bl = np.vectorize(belong)
belong = func_bl(Q9[0], Q9[1], Q9[2])
Q9['belong'] = belong

# safe
def safe(x, y, z):
    if x == 'I feel safe on SLCC campuses.':
        return 1
    elif y == 'I feel safe on SLCC campuses.':
        return 1
    elif z == 'I feel safe on SLCC campuses.':
        return 1
    else:
        return 0

func_safe = np.vectorize(safe)
safe = func_safe(Q9[0], Q9[1], Q9[2])
Q9['safe'] = safe

# Q8 and Q7 -> ordered factors

def q8q7(x):
    if x == 'Very comfortable':
        return 5
    elif x == 'Somewhat comfortable':
        return 4
    elif x == 'Neither comfortable nor uncomfortable':
        return 3
    elif x == 'Somewhat uncomfortable':
        return 2
    elif x == 'Very uncomfortable':
        return 1
    else: 
        return 0
    
func_q8q7 = np.vectorize(q8q7)
slcc_comfort = func_q8q7(data["Q7"])
data["slcc_comfort"] = slcc_comfort

div_comfort = func_q8q7(data["Q8"])
data["div_dept_comfort"] = div_comfort

# Q10 -> ordered factor
def q10(x):
    if x == 'Improving':
        return 3
    elif x == 'Staying the same':
        return 2
    elif x == 'Getting worse':
        return 1
    else:
        return 0

func_q10 = np.vectorize(q10)
improvement = func_q10(data["Q10"])
data["improvement"] = improvement

# 
data = pd.concat([data, Q9], axis = 1)

# Q12 dummy
data = pd.concat([data, pd.get_dummies(data['Q12'], prefix='excluded')], axis = 1)


# limiting the primary data set to the variables of interest

data_rd = data[["slcc_comfort", "div_dept_comfort",
                "welcome", "belong", "safe", "improvement", 
                "excluded_Yes", "Q2", "Q6", "Q43", 
                "Q44", "Q45", "Q50", "Q55", "Q48",
                "Q46", "Q47", "Q51", "status"]] 

data_rd = data_rd.fillna("No Answer")

data_rd.to_csv("cs2019_data_cleaned.csv")
data_key.to_csv("cs2019_data_key.csv")
data.to_csv("cs2019_full_clean.csv") # cleaned variables with all other variables
