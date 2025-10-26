import os
import math
import pandas as pd
import random
import locale
import warnings
import seaborn
import numpy as np
import tkinter as tk
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from sklearn.neighbors import LocalOutlierFactor
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

warnings.filterwarnings("ignore")


# Attempting again with the newly uploaded file
data = pd.read_csv("/Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/DataBaseTotal.csv")

noms_distincts = list(data["Nom"].unique())
N = len(noms_distincts)

# Process the data as before
data_complet = data.dropna()
data_complet['Distance'] = data_complet['Distance'].str.replace(',', '.', regex=False)
data_complet['Distance'] = pd.to_numeric(data_complet['Distance'], errors='coerce')
data_complet = data_complet.reset_index(drop=True)
data_complet = data_complet[data_complet['Distance'] > 2]
data_complet['Temps en minutes'] = data_complet['Durée de déplacement'] / 60

# Remove unnecessary columns
data_course = data_complet.drop(columns=['Durée de déplacement', "Nom", "ID de l'activité", 
                                         "Date de l'activité", "Type d'activité", 
                                         "Nom du fichier"])

data_course["nom"] = data_complet["Nom"]

# Initialize cleaned DataFrames for storing results
data_cleaned_loop = pd.DataFrame()
data_cleaned_complet = pd.DataFrame()

# Apply LOF for each person to remove outliers
unique_names = data_course["nom"].unique()

for name in unique_names:
    person_data = data_course[data_course["nom"] == name]
    person_datacomplet = data_complet[data_complet["Nom"] == name]

    person_features = person_data.iloc[:, :-1]
    
    lof = LocalOutlierFactor(n_neighbors=10, algorithm='auto', metric='euclidean', contamination=0.15)
    labels = lof.fit_predict(person_features)
    
    person_data["LOF_Label"] = labels
    person_datacomplet["LOF_Label"] = labels
    
    person_datacomplet = person_datacomplet[person_datacomplet["LOF_Label"] == 1].drop(columns=["LOF_Label"])
    person_cleaned = person_data[person_data["LOF_Label"] == 1].drop(columns=["LOF_Label"])
    
    data_cleaned_loop = pd.concat([data_cleaned_loop, person_cleaned], ignore_index=True)
    data_cleaned_complet = pd.concat([data_cleaned_complet, person_datacomplet], ignore_index=True)



# Reorganize the cleaned dataset
data_course_CR = data_cleaned_loop.drop(columns=["nom"])
data_course_CR["Nom"] = data_cleaned_complet["Nom"]
data_course_CR["ID de l'activité"] = data_cleaned_complet["ID de l'activité"]
data_course_CR["Date de l'activité"] = data_cleaned_complet["Date de l'activité"]

# Calculate average speed (km/h) and format the date
data_course_CR['Vitesse Moyenne (min/km)'] =  data_cleaned_complet['Temps en minutes'] / (data_course_CR['Distance ajustée selon la pente'] / 1000)




# Créer une figure unique
figEvo = plt.figure(figsize=(12, 8))

# Parcourir chaque personne unique dans les données
for name in data_course_CR['Nom'].unique():
    # Filtrer les données pour chaque personne
    person_subset = data_course_CR[data_course_CR['Nom'] == name]
    
    # Calculer la moyenne et l'écart-type
    mean_speed = person_subset['Vitesse Moyenne (min/km)'].mean()
    std_dev_speed = person_subset['Vitesse Moyenne (min/km)'].std()
    
    # Calculer les limites de l'intervalle
    lower_bound = mean_speed - 2.96 * std_dev_speed
    upper_bound = mean_speed + 2.96 * std_dev_speed
    
    # Filtrer les points à l'intérieur de l'intervalle
    filtered_subset = person_subset[
        (person_subset['Vitesse Moyenne (min/km)'] >= lower_bound) &
        (person_subset['Vitesse Moyenne (min/km)'] <= upper_bound)
    ]
    
    # Tracer la courbe pour cette personne
    plt.plot(filtered_subset['Distance'], 
             filtered_subset['Vitesse Moyenne (min/km)'], 
             marker='o', label=f"{name} (Vitesse Moyenne)")
    
# Ajouter les titres et les légendes
plt.title("Évolution de la vitesse moyenne pour chaque personne selon les Km")
plt.xlabel("Distance (km)")
plt.ylabel("Vitesse Moyenne (min/km)")
plt.grid(True)
plt.legend()
figEvo.savefig("/Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/Evo-Pace-Selon-Distance/Evo-Pace-Distance.png", format='png', dpi=300)
plt.show()


