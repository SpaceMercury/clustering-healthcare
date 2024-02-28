# -*- coding: utf-8 -*-
"""clustering.ipynb pol

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1kT1e8OUWNr-35PawUu-Rc22_eGXqh0pc
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn import preprocessing
from kmodes.kmodes import KModes

# Load the data from a local file
file_path = "CESR.CSV"
raw_data = pd.read_csv(file_path)
dataframe = pd.DataFrame(raw_data)


names = {"V02" : "Sex",
         "V03" : "Nationality",
         "V21" : "Location of accident",
         "V22" : "Traffic accident",
         "V34" : "Time of day",
         "V35" : "Hour of shift",
         "V38" : "Location type",
         "V39" : "Type of work",
         "V44" : "Cause of injury",
         "V47" : "Injury description",
         "V48" : "Severity of injury",
         "V57" : "Age"}
dataframe = dataframe.rename(columns = names)

def prepsex(sex):
    if sex == 1:
        return "male"
    elif sex == 2:
        return "female"
    else:
        raise Exception()
dataframe["Sex"] = dataframe["Sex"].apply(prepsex)

def prepnat(nat):
    if nat == 724:
        return "Spanish"
    else:
        return "non-Spanish"

dataframe["Nationality"] = dataframe["Nationality"].apply(prepnat)

dataframe = dataframe[dataframe["Location of accident"] != 3]
def preploc(loc): # translates location in context of work
    if loc == 1:
        return "in usual workplace"
    elif loc == 2:
        return "on assignment"
    elif loc == 3:
        return "on way to or from workplace"
    elif loc == 4:
        return "other workplace"
    else:
        raise Exception()
dataframe["Location of accident"] = dataframe["Location of accident"].apply(preploc)

def preptime(time):
    if time < 4:
        return "late night"
    elif time < 8:
        return "early morning"
    elif time < 12:
        return "late morning"
    elif time < 16:
        return "afternoon"
    elif time < 20:
        return "evening"
    elif time < 25:
        return "early night"
    else:
        raise Exception()
dataframe["Time of day"] = dataframe["Time of day"].apply(preptime)

def prephour(hour):
    if hour == 0:
        return "going to work"
    elif hour == 99:
        return "coming from work"
    elif hour <= 3:
        return "1-3"
    elif hour <= 6:
        return "4-6"
    elif hour <= 9:
        return "7-9"
    elif hour <= 12:
        return "10-12"
    else:
        return "13+"

dataframe["Hour of shift"] = dataframe["Hour of shift"].apply(prephour)

def preptype(typ): # categorizes location type
    translation = {(10, 11, 12, 13, 19) : "Industrial - L",
                   (20, 21, 22, 23, 24, 25, 26, 29) : "Construction - L",
                   (30, 31, 32, 33, 34, 35, 36, 39) : "Agricultural - L",
                   (40, 41, 42, 43, 44, 49) : "Commercial - L",
                   (50, 51, 59) : "Clinical - L",
                   (60, 61, 62, 69, 93, 101, 110, 111, 112, 119) : "Transportation - L",
                   (63,) : "Transportation, authorized personnel only - L",
                   (70, 71, 72, 79) : "Residential - L",
                   (80, 81, 82, 89) : "Sporting facility - L",
                   (90, 91, 92, 99) : "In the air - L",
                   (100, 102, 103, 109) : "Underground - L",
                   (120, 121, 122, 129) : "Underwater - L",
                   (0, 999) : "Little/no information - L"}
    for codes in translation:
        if typ in codes:
            return translation[codes]
    raise Exception()
dataframe["Location type"] = dataframe["Location type"].apply(preptype)

def prepwork(work):
    translation = {(10, 11, 12, 19) : "Manufacturing - W",
                   (20, 21, 22, 23, 24, 25, 29) : "Construction - W",
                   (30, 31, 32, 33, 34, 35, 39) : "Agriculture - W",
                   (41,) : "Services and healthcare - W",
                   (42,) : "Intellectual - W",
                   (43,) : "Sales - W",
                   (51, 52) : "Installation and maintenance - W",
                   (53, 54) : "Sanitation - W",
                   (55,) : "Inspection - W",
                   (61,) : "Transportation - W",
                   (62,) : "Arts and sports - W",
                   (0, 40, 49, 50, 59, 60, 69, 99) : "Little/no information - W"}
    for codes in translation:
        if work in codes:
            return translation[codes]
    print(work)
    raise Exception()

dataframe["Type of work"] = dataframe["Type of work"].apply(prepwork)

def prepcause(cause):
    translation = {(11, 12) : "Electricity",
                   (13, 14) : "Extreme temperatures",
                   (15, 16, 17) : "Dangerous substances",
                   (21,) : "Drowning",
                   (22,) : "Burying alive",
                   (23,) : "Gases or aerosols",
                   (30, 31, 32, 39) : "Immobile object",
                   (40, 41, 42, 43, 44, 45, 46, 49) : "Mobile object",
                   (50, 51, 52, 53, 59) : "Sharp object",
                   (60, 61, 62, 63, 64, 69) : "Crushing and amputation",
                   (71,) : "Overexertion",
                   (72,) : "Radiation, noise, light, or pressure",
                   (73,) : "Psycological trauma",
                   (80, 81, 82, 83, 89) : "Bites and kicks",
                   (90,) : "Non-traumatic injuries",
                   (0, 10, 19, 20, 29, 70, 79, 99) : "Little/no information - C"}
    for codes in translation:
        if cause in codes:
            return translation[codes]
    print(cause)
    raise Exception()
dataframe["Cause of injury"] = dataframe["Cause of injury"].apply(prepcause)

def prepdesc(desc):
    translation = {(10, 11, 12, 19) : "Superficial lesions",
                   (20, 21, 22, 29) : "Fractures",
                   (30, 31, 32, 39) : "Dislocations, sprains, and pulled muscles",
                   (40,) : "Amputations",
                   (50, 51, 52, 59) : "Internal injuries",
                   (61, 62) : "Burn",
                   (63,) : "Frostbite",
                   (70, 71, 72, 79) : "Poison and infection",
                   (80, 81, 82, 89) : "Asphixia",
                   (91, 92) : "Loss of hearing and barotrauma",
                   (101,) : "Heatstroke",
                   (102,) : "Radiation damage",
                   (103,) : "Hypothermia",
                   (111,) : "Psychological damage",
                   (112, 119) : "Shock",
                   (120,) : "Multiple injuries",
                   (130,) : "Non-traumatic injury",
                   (0, 60, 69, 90, 99, 100, 109, 110, 999) : "Little/no information - D"}
    for codes in translation:
        if desc in codes:
            return translation[codes]
    print(desc)
    raise Exception()
dataframe["Injury description"] = dataframe["Injury description"].apply(prepdesc)

def prepsev(sev):
    if sev == 1:
        return "mild"
    elif sev == 2:
        return "severe"
    elif sev == 3:
        return "very severe"
    elif sev == 4:
        return "fatal"
    else:
        print(sev)
        raise Exception()
dataframe["Severity of injury"] = dataframe["Severity of injury"].apply(prepsev)

def prepage(age):
    if age < 18:
        return "under 18"
    elif age < 20:
        return "18-19"
    elif age < 30:
        return "20-29"
    elif age < 40:
        return "30-39"
    elif age < 50:
        return "40-49"
    elif age < 60:
        return "50-59"
    elif age < 70:
        return "60-69"
    elif age < 80:
        return "70-79"
    else:
        return "80+"
dataframe["Age"] = dataframe["Age"].apply(prepage)

dataframe.groupby("Cause of injury").size().plot(kind="bar")
##Group all of the data by the cause of injury and count how many instances of each

# initialization, creation of the 64 element list of permutations of our 7 variables

variables = ["Location of accident", "Time of day", "Hour of shift", "Cause of injury", "Injury description", "Severity of injury", "Age"]
variable_combinations_list = []
for i in range(4, len(variables) + 1):
    variable_combinations_list.extend(combinations(variables, i))
variable_combinations_list = [list(combination) for combination in variable_combinations_list]
variable_combinations_list = variable_combinations_list[::-1]

table = pd.DataFrame()

print(len(variable_combinations_list))

"""

Generates lists of 4,5,6,7 variables in
one list choosing from the variables list
Combination of these variables, no repetition
Outputs a list of lists size 64 with these combinations inside

"""

# Iterate over combinations
for combination in variable_combinations_list[64:65]:

  print(f"Now working on {len(variable_combinations_list)}")
  if len(variable_combinations_list) % 10 == 0:
      print(f"{len(variable_combinations_list)} more to go")

  # Select columns from 'variables' DataFrame based on the combination
  variable_combinations_listubgroup = dataframe[combination]
  # Clustering code

  ## Preprocess data, apply label encoding to subgroup
  ## 
  subgroup = variable_combinations_listubgroup.copy()
  from sklearn import preprocessing
  le = preprocessing.LabelEncoder()
  subgroup_encoded = subgroup.apply(le.fit_transform)


  # Iterate through cluster numbers
  for n_clusters in range(3, 11):
      # Clustering
      km = KModes(n_clusters=n_clusters, init='Cao', n_init=1, verbose=0)
      clusters = km.fit_predict(subgroup_encoded)

      # Combine with original data
      # adds column of the clusters that have calculated by kmodes
      # instead of using .drop, we could just set drop = true in subgroup.reset_index
      combined = pd.concat([subgroup.reset_index(), pd.DataFrame({'Cluster': clusters})], axis=1).drop('index', axis=1)
      table = pd.DataFrame(columns=range(n_clusters))

      # Fill in table with percentage of instances of each value in each cluster for each variable

      for var in subgroup.columns:
          # counts how many instances of a given variable are in the cluster
          # for example out of x data points in cluster 1, 500 were aged between 60-69
          by_cluster_value = combined.groupby(['Cluster', var], group_keys=False).size()
          percentages = by_cluster_value.groupby(level=0, group_keys=False).apply(lambda x: x / x.sum() * 100)
          pivtable = pd.pivot_table(percentages.reset_index(), index=var, columns='Cluster', values=0, fill_value=0)


          # Fill in table with percentage of instances for each value in each cluster
          for i, col in enumerate(pivtable.columns):
              values = pivtable[col].index.tolist()
              pct_values = pivtable[col].tolist()
              for j, value in enumerate(values):
                  table.loc[value, i] = pct_values[j]
          # Add new column with variable names based on index
      table['Variable'] = ''
      variable_mapping = {}

      #finishing up table, adds columns for variable and value of variable
      for column in subgroup.columns:
          unique_values = tuple(subgroup[column].unique())
          variable_mapping[unique_values] = column

      for i, index_value in enumerate(table.index.values):
          for values in variable_mapping:
              if index_value in values:
                  table.iloc[i, -1] = variable_mapping[values]
                  break

      table = table.reset_index().rename(columns={'index': 'Value'})
      table = table[['Variable'] + list(table.columns[:-1])]
      #print(table)
      table = table.set_index('Variable')

      # Add a new row for the cluster drivers percentage
      table.loc['Percentage of accidents'] = 0

      # Calculate the number of instances in each cluster
      cluster_counts = combined['Cluster'].value_counts()

      # Fill in the table with the percentage of instances in each cluster
      for i, count in cluster_counts.items():
          table.loc['Percentage of accidents', i] = count / len(combined) * 100

      #table.to_excel(f'table_{n_clusters}_clusters_pas1.xlsx', index=True)
      table1 = table
      # Save table to CSV file
      table = table.drop('Value', axis=1)
      new_table_2 = pd.DataFrame(index=table.index.unique())
  # Iterate over each column (0, 1, 2, 3) and find the highest percentage for each variable
      for col in table.columns:
          col_max = table.groupby(level=0, group_keys=False)[col].max()
          new_table_2[col] = col_max
  # Add the 'Percentage of accidents' row back to the new DataFrame
      new_table_2.loc['Percentage of accidents'] = table.loc['Percentage of accidents']




  # Display the new DataFrame
      new_table_3= new_table_2.loc[new_table_2.index != 'Percentage of accidents'] = (new_table_2.loc[new_table_2.index != 'Percentage of accidents'] >= 75).astype(int)
      new_table_3.loc['Variables => 75%'] = new_table_3.sum()
      new_table_3.loc['Cluster Value'] = new_table_3 .loc['Variables => 75%'] / variable_combinations_listubgroup.shape[1]
      new_table_3.loc['Cluster adequate?'] = new_table_3 .loc['Cluster Value'] > 0.5
      new_table_4=new_table_3
      new_table_4.loc['Percentage of accidents'] = table.loc['Percentage of accidents']
      new_table_4.loc['Score'] = np.nan
      new_table_4.loc['Score' ,'Total clusterization value'] = new_table_4.loc['Cluster adequate?'].sum() / n_clusters
      ##print(new_table_4)
      if new_table_4.loc['Cluster adequate?'].sum() / n_clusters > 0.9:
          ##print(table1)
          print("Cluster Found!!!")
          table1.to_excel(f'Output/table_{n_clusters}_clusters_{"".join(combination)}.xlsx', index=True)


  variable_combinations_list.remove(combination)

