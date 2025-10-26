import os
import math
import pandas as pd
import random
import locale
import seaborn
import numpy as np
import tkinter as tk
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from sklearn.neighbors import LocalOutlierFactor
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# Dictionnaire pour remplacer les mois abrégés en français par leur forme complète
mois_francais = {
    "janv.": "janvier",
    "févr.": "février",
    "mars": "mars",
    "avr.": "avril",
    "mai": "mai",
    "juin": "juin",
    "juil.": "juillet",
    "août": "août",
    "sept.": "septembre",
    "oct.": "octobre",
    "nov.": "novembre",
    "déc.": "décembre"
}


# Normaliser les dates
def normaliser_mois(date):
    for mois_abrege, mois_complet in mois_francais.items():
        date = date.replace(mois_abrege, mois_complet)
    return date


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

#locale.setlocale(locale.LC_TIME, "fr_FR.UTF-8")

# Reorganize the cleaned dataset
data_course_CR = data_cleaned_loop.drop(columns=["nom"])
data_course_CR["Nom"] = data_cleaned_complet["Nom"]
data_course_CR["ID de l'activité"] = data_cleaned_complet["ID de l'activité"]
data_course_CR["Date de l'activité"] = data_cleaned_complet["Date de l'activité"]

# Calculate average speed (km/h) and format the date
data_course_CR['Vitesse Moyenne (km/h)'] =  data_cleaned_complet['Temps en minutes'] / (data_course_CR['Distance ajustée selon la pente'] / 1000)

# Remove the "à" and time part from the date strings
data_course_CR['Cleaned Date'] = data_course_CR['Date de l\'activité'].str.split(' à ').str[0]



data_course_CR['Normalized Date'] = data_course_CR['Cleaned Date'].apply(normaliser_mois)

# Conversion des dates normalisées en format datetime
data_course_CR['Parsed Date'] = pd.to_datetime(data_course_CR['Normalized Date'], format='%d %B %Y', errors='coerce', dayfirst=True)

# Formater les dates dans le format souhaité (dd/mm/yyyy)
data_course_CR['Formatted Date'] = data_course_CR['Parsed Date'].dt.strftime('%d/%m/%Y')

#data_course_CR = data_course_CR.drop(columns=['Cleaned Date','Parsed Date'])


# Tri chronologique par Formatted Date pour chaque personne
person_data_cat = pd.DataFrame()
person_tables = {}
for name in data_course_CR["Nom"].unique():
    person_data = data_course_CR[data_course_CR["Nom"] == name][['Normalized Date', 'Distance ajustée selon la pente', "Vitesse Moyenne (km/h)"]]
    person_data["Nom"] = name
    person_tables[name] = person_data.sort_values(by='Normalized Date',axis = 0, ascending=True)
    person_data_cat = pd.concat([person_data_cat,person_data],ignore_index=True)
    
person_data_cat.to_csv('/Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/person_data_cat.csv', index=False)


# Fenêtre Tkinter
rootPace = tk.Tk()
rootPace.title("Graphique t-SNE")

# Nombre de graphiques
N = len(person_data_cat['Nom'].unique())
rows = int(np.ceil(N / 4))  # Nombre de lignes
cols = min(4, N)  # Nombre de colonnes, max 4 par ligne

# Créer la figure Matplotlib
figPace, axes = plt.subplots(rows, cols, figsize=(15, 10))

# Si la grille est 1D, convertir `axes` en une liste pour une itération facile
axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

# Tracer les graphiques pour chaque personne
for idx1, name in enumerate(person_data_cat['Nom'].unique()):
    # Filtrer les données pour chaque personne
    person_subset = person_data_cat[person_data_cat['Nom'] == name].sort_values(by='Normalized Date')
    
    # Calculer la moyenne et l'écart-type
    mean_speed = person_subset['Vitesse Moyenne (km/h)'].mean()
    std_dev_speed = person_subset['Vitesse Moyenne (km/h)'].std()
    
    # Calculer les limites de l'intervalle
    lower_bound = mean_speed - 2.96 * std_dev_speed
    upper_bound = mean_speed + 2.96 * std_dev_speed
    
    # Filtrer les points à l'intérieur de l'intervalle
    filtered_subset = person_subset[
        (person_subset['Vitesse Moyenne (km/h)'] >= lower_bound) &
        (person_subset['Vitesse Moyenne (km/h)'] <= upper_bound)
    ]
    
    # Tracer sur l'axe correspondant
    ax = axes[idx1]
    ax.plot(filtered_subset['Normalized Date'], filtered_subset['Vitesse Moyenne (km/h)'], 
            marker='o', label="Vitesse Moyenne (filtrée)")
    ax.axhline(mean_speed, color='red', linestyle='--', label=f"Moyenne ({mean_speed:.2f})")
    ax.fill_between(filtered_subset['Normalized Date'], 
                    mean_speed - std_dev_speed, 
                    mean_speed + std_dev_speed, 
                    color='orange', alpha=0.2, label="± Écart-type")
    
    # Configuration de l'axe X pour respecter l'espacement temporel
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(DateFormatter("%d/%m/%Y"))
    ax.tick_params(axis='x', rotation=45)
    
    # Ajouter le titre et les légendes
    ax.set_title(f"Évolution pour {name}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Vitesse Moyenne (km/h)")
    ax.grid(True)
    ax.legend()
    
   # fig3D.savefig("/Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/resultats t-SNE/Plot-tSNE/Projection des données 3D.png", format='png', dpi=300)  # `dpi=300` pour une haute qualité


# Masquer les sous-graphiques inutilisés (si N ne remplit pas toute la grille)
for i in range(N, len(axes)):
    axes[i].axis('off')  # Cacher le sous-graphe inutilisé

# Ajuster l'espacement entre les sous-graphiques
plt.tight_layout()

# Intégrer la figure Matplotlib dans Tkinter
canvasPace = FigureCanvasTkAgg(figPace, master=rootPace)
canvasPace.draw()
canvasPace.get_tk_widget().pack(expand=True, fill="both")

# Ajouter une barre d'outils de navigation
toolbarPace = NavigationToolbar2Tk(canvasPace, rootPace)
toolbarPace.update()
toolbarPace.pack(side=tk.BOTTOM, fill=tk.X)

# Lancer l'application Tkinter
rootPace.mainloop()


