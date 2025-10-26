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
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
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

# Initialiser une liste pour stocker les résultats
regression_results = []

#####---------------------LINEAR-----------------------########


# # Parcourir chaque personne unique dans les données
# for name in data_course_CR['Nom'].unique():
#     # Filtrer les données pour chaque personne
#     person_subset = data_course_CR[data_course_CR['Nom'] == name]
    
#     # Calculer la moyenne et l'écart-type
#     mean_speed = person_subset['Vitesse Moyenne (min/km)'].mean()
#     std_dev_speed = person_subset['Vitesse Moyenne (min/km)'].std()
    
#     # Calculer les limites de l'intervalle
#     lower_bound = mean_speed - 2.96 * std_dev_speed
#     upper_bound = mean_speed + 2.96 * std_dev_speed
    
#     # Filtrer les points à l'intérieur de l'intervalle
#     filtered_subset = person_subset[
#         (person_subset['Vitesse Moyenne (min/km)'] >= lower_bound) &
#         (person_subset['Vitesse Moyenne (min/km)'] <= upper_bound)
#     ]
    
#     # Extraire les données pour la régression linéaire
#     x = filtered_subset['Distance'].values.reshape(-1, 1)  # Distance (variable indépendante)
#     y = filtered_subset['Vitesse Moyenne (min/km)'].values  # Vitesse Moyenne (variable dépendante)
    
#     # Vérifier qu'il y a suffisamment de points pour effectuer une régression
#     if len(x) > 1:
#         # Appliquer la régression linéaire
#         model = LinearRegression()
#         model.fit(x, y)
#         y_pred = model.predict(x)  # Prédictions pour les points existants
        
#         # Calculer les coefficients
#         slope = model.coef_[0]
#         intercept = model.intercept_
#     else:
#         slope, intercept = None, None  # Cas avec trop peu de points pour une régression

#     # Créer une nouvelle figure pour chaque personne
#     plt.figure(figsize=(12, 8))
    
#     # Tracer le nuage de points
#     plt.scatter(filtered_subset['Distance'], 
#                 filtered_subset['Vitesse Moyenne (min/km)'], 
#                 color='blue', alpha=0.6, label="Points filtrés")
    
#     # Tracer la ligne de régression
#     if slope is not None:
#         plt.plot(filtered_subset['Distance'], y_pred, color='red', 
#                  label=f"Ligne de régression (y = {slope:.2f}x + {intercept:.2f})")
    
#     # Ajouter les titres et les légendes
#     plt.title(f"Évolution de la vitesse moyenne pour {name} selon les Km (Régression incluse)")
#     plt.xlabel("Distance (km)")
#     plt.ylabel("Vitesse Moyenne (min/km)")
#     plt.grid(True)
#     plt.legend()
    
#     # Sauvegarder le graphique pour cette personne
#     plt.savefig(f"/Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/Evo-Pace-Selon-Distance/Evo-Pace-DistanceLinear-{name}.png", format='png', dpi=300)
    
#     # Afficher le graphique
#     plt.show()

#####---------------------POLYNOMIAL-----------------------########

# Parcourir chaque personne unique dans les données
for name in data_course_CR['Nom'].unique():
    # Filtrer les données pour chaque personne
    
    person_subset = data_course_CR[data_course_CR['Nom'] == name]
    
    # Extraire les données pour la régression
    x = person_subset['Distance'].values.reshape(-1, 1)  # Distance (variable indépendante)
    y = person_subset['Vitesse Moyenne (min/km)'].values  # Vitesse Moyenne (variable dépendante)
    
    y_mean = np.mean(y)
    y_std = np.std(y)
    y_normalized = ((y - y_mean) / y_std)/5 + 1
    
   # Vérifier qu'il y a suffisamment de points pour effectuer une régression
    if len(x) > 2:
        # Créer un pipeline pour la régression polynomiale
        degree = 2  # Choisissez le degré du polynôme
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(x, y_normalized)
        y_pred = model.predict(x)
        
        # Extraire les coefficients et l'intercept
        poly_features = model.named_steps['polynomialfeatures']  # Étape pour générer les termes polynomiaux
        linear_model = model.named_steps['linearregression']  # Étape du modèle linéaire
        
        coefficients = linear_model.coef_  # Coefficients des termes polynomiaux
        intercept = linear_model.intercept_  # Intercept du modèle
        
        # Afficher les coefficients
        print(f"Régression polynomiale pour {name} (degré {degree}):")
        print(f"Intercept: {intercept:.2f}")
        for i, coef in enumerate(coefficients):
            print(f"Coefficient du terme x^{i}: {coef:.2f}")
        print("\n")
        
        # Ajouter les résultats dans la liste
        regression_results.append({
            "Nom": name,
            "Intercept": intercept,
            **{f"Coefficient_x^{i}": coef for i, coef in enumerate(coefficients)}
        })
    else:
        # Ajouter une ligne vide si pas assez de données pour cette personne
        regression_results.append({
            "Nom": name,
            "Intercept": None,
            **{f"Coefficient_x^{i}": None for i in range(degree + 1)}
        })

    # Créer une nouvelle figure pour chaque personne
    plt.figure(figsize=(12, 8))
    
    # Convertir les résultats en DataFrame pour un affichage tabulaire
    results_df = pd.DataFrame(regression_results)
    
    # Tracer le nuage de points
    plt.scatter(x, y_normalized, color='blue', alpha=0.6, label="Points réels")
    
    # Tracer la courbe de régression polynomiale
    if y_pred is not None:
        plt.scatter(x, y_pred, color='red', label=f"Régression polynomiale (degré {degree})")
    
    # Ajouter les titres et les légendes
    plt.title(f"Évolution de la vitesse moyenne pour {name} avec régression polynomiale")
    plt.xlabel("Distance (km)")
    plt.ylabel("Vitesse Moyenne (min/km)")
    plt.grid(True)
    plt.legend()
    
    # Sauvegarder le graphique pour cette personne
    plt.savefig(f"/Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/Evo-Pace-Selon-Distance/Evo-Pace-PolyReg-{name}.png", format='png', dpi=300)
    results_df.to_csv("/Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/Evo-Pace-Selon-Distance/Regression_Coefficients.csv", index=False)

    # Afficher le graphique
    plt.show()


