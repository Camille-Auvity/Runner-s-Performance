import os
import pandas as pd
import numpy as np
import tkinter as tk
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from sklearn.manifold import TSNE
from sklearn.neighbors import LocalOutlierFactor

import locale
import warnings

from datetime import datetime
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from statsmodels.tsa.statespace.sarimax import SARIMAX

#------------------------------------------------------------------------------
# Projet Industriel
#------------------------------------------------------------------------------

Chemin_Dossier = '/Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/Data_course'
os.chdir(Chemin_Dossier)
fichiers = os.listdir()
fichiers.pop(1)


Colonnes_a_garder = [
    'Nom','ID de l\'activité', 'Date de l\'activité', 'Type d\'activité', 'Temps écoulé', 
    'Distance', 'Fréquence cardiaque maximum', 'Mesure d\'effort', 'Nom du fichier', 
    'Poids de l\'athlète', 'Vitesse max.', 'Vitesse moyenne', 'Dénivelé positif', 
    'Dénivelé négatif', 'Altitude min.', 'Altitude max.', 'Pente max.', 'Pente moyenne', 
    'Fréquence cardiaque maximum', 'Fréquence cardiaque moyenne', 'Calories', 'Température max.', 
    'Température moyenne', 'Heure d\'observation de la météo', 'Conditions météo', 
    'Température selon les prévisions météo', 'Température ressentie', 'Point de rosée', 
    'Humidité', 'Pression atmosphérique', 'Vitesse du vent', 'Rafale de vent', 
    'Direction du vent', 'Intensité des précipitations', 'Probabilité de précipitations', 
    'Type de précipitations', 'Couverture nuageuse', 'Indice UV', 
    'Ozone selon les prévisions météo']

Colonnes_a_garder2 = [
    'Nom','ID de l\'activité', 'Date de l\'activité', 'Type d\'activité', 'Durée de déplacement', #Temps écoulé', 
    'Distance', 'Nom du fichier', 'Vitesse max.', 'Vitesse moyenne', 'Dénivelé positif', 
    'Dénivelé négatif', 'Pente moyenne', 'Distance ajustée selon la pente']


Colonnes_a_garder3 = [94, 0, 1, 3, 16, 6, 12, 18, 19, 20, 21, 25, 53]


Data = pd.DataFrame(columns=Colonnes_a_garder2)

# Charger le fichier CSV

for i in fichiers:
    prenom = i.split('_')[1].split('.')[0]
    
    df = pd.read_csv(f'/Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/Data_course/{i}')

    df["Nom"] = f'{prenom}'
    
    new_df = df.iloc[:, Colonnes_a_garder3]
    
    for i in range(new_df.shape[0]):
        if new_df[new_df.columns[3]][i] == 'Run':
            new_df[new_df.columns[3]][i] = 'Course à pied'

    new_df.columns = ['Nom', 'ID de l\'activité', 'Date de l\'activité', 'Type d\'activité', 'Durée de déplacement', 
                      'Distance', 'Nom du fichier', 'Vitesse max.', 'Vitesse moyenne', 'Dénivelé positif', 
                      'Dénivelé négatif', 'Pente moyenne', 'Distance ajustée selon la pente']
    
    # Filtrer pour ne conserver que les lignes où le type d'activité est "Course à pied"
    df_filtre = new_df[new_df[new_df.columns[3]] == 'Course à pied']

    # Ajouter les données filtrées au DataFrame principal
    Data = pd.concat([Data, df_filtre], ignore_index=True)

# Supprimer la ligne où ID == 5241179994
Data = Data[Data['ID de l\'activité'] != 5247153691]

# Réinitialiser les index
Data.reset_index(drop=True, inplace=True)

Data.to_csv('/Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/DataBaseTotal.csv', index=False)

#------------------------------------------------------------------------------
# PI - TSNE
#------------------------------------------------------------------------------


data = pd.read_csv("/Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/DataBaseTotal.csv")

noms_distincts = list(data["Nom"].unique())
N = len(noms_distincts)

data_complet = data.dropna()
data_complet['Distance'] = data_complet['Distance'].str.replace(',', '.', regex=False)
data_complet['Distance'] = pd.to_numeric(data_complet['Distance'], errors='coerce')
data_complet = data_complet.reset_index(drop=True)
data_complet = data_complet[data_complet['Distance'] > 2]
data_complet['Temps en minutes'] = data_complet['Durée de déplacement'] / 60

data_course = data_complet.drop(columns=['Durée de déplacement',"Nom", "ID de l'activité", 
                                         "Date de l'activité", "Type d'activité", "Nom du fichier",
                                         "Distance ajustée selon la pente"])

mean = data_course.mean()  # Moyenne par colonne
std = data_course.std()    # Écart-type par colonne
data_course_CR_wOLnom = (data_course - mean) / std

#______________________Detection_outliers____________________________

data_course_CR_wOLnom["nom"] = data_complet["Nom"]

# Initialiser un DataFrame vide pour stocker les données filtrées
data_cleaned_loop = pd.DataFrame()
data_cleaned_complet = pd.DataFrame()


# Obtenir les noms uniques (personnes)
unique_names = data_course_CR_wOLnom["nom"].unique()

# Parcourir chaque personne
for name in unique_names:
    # Filtrer les données pour la personne actuelle
    person_data = data_course_CR_wOLnom[data_course_CR_wOLnom["nom"] == name]
    person_datacomplet = data_complet[data_complet["Nom"] == name]

    # Sélectionner uniquement les colonnes de caractéristiques (exclure "nom")
    person_features = person_data.iloc[:, :-1]
    
    # Appliquer LOF pour cette personne
    lof = LocalOutlierFactor(n_neighbors=10, algorithm='auto', metric='euclidean', contamination=0.15)
    labels = lof.fit_predict(person_features)
    
    # Ajouter les données sans outliers au DataFrame final
    person_data["LOF_Label"] = labels  # Ajouter les labels (-1 pour outliers, 1 pour points normaux)
    person_datacomplet["LOF_Label"] = labels
    
    person_datacomplet = person_datacomplet[person_datacomplet["LOF_Label"] == 1].drop(columns=["LOF_Label"])
    person_cleaned = person_data[person_data["LOF_Label"] == 1].drop(columns=["LOF_Label"])  # Points normaux seulement
    data_cleaned_loop = pd.concat([data_cleaned_loop, person_cleaned], ignore_index=True)
    data_cleaned_complet = pd.concat([data_cleaned_complet, person_datacomplet], ignore_index=True)

# Vérifier les dimensions avant et après la suppression des outliers
initial_points_loop = len(data_course_CR_wOLnom)
final_points_loop = len(data_cleaned_loop)
initial_points_loop, final_points_loop

data_course_CRnom = data_cleaned_loop.copy()

data_course_CR = data_cleaned_loop.drop(columns=["nom"])


# ________________Application et évaluation du t-SNE___________________________

tsne = TSNE(n_components=3, perplexity=30, learning_rate=200, random_state=42)
data_tsne = tsne.fit_transform(data_course_CR)

# Transformation des données en DataFrame
data_tsne_df = pd.DataFrame({
    "Dim1": data_tsne[:, 0],
    "Dim2": data_tsne[:, 1],
    "Dim3": data_tsne[:, 2],
    'Temps en minutes': data_course_CR['Temps en minutes'],
    "Distance": data_course_CR["Distance"],
    "ID de l'activité": data_cleaned_complet["ID de l'activité"],
    "Nom": data_cleaned_complet["Nom"]
})

from sklearn.manifold import trustworthiness

# Calcul de la Trustworthiness
trust = trustworthiness(data_course_CR, data_tsne, n_neighbors=10)
print(f"Trustworthiness: {trust:.4f}")

from sklearn.metrics import pairwise_distances

def continuity(original_data, embedded_data, n_neighbors=10):
    # Calcul des distances
    original_distances = pairwise_distances(original_data)
    embedded_distances = pairwise_distances(embedded_data)
    
    # Rangs des distances
    original_ranks = np.argsort(np.argsort(original_distances, axis=1), axis=1)
    embedded_ranks = np.argsort(np.argsort(embedded_distances, axis=1), axis=1)
    
    # Différence des voisins proches
    n = original_data.shape[0]
    continuity_score = 1 - (2 / (n * n_neighbors * (2 * n - 3 * n_neighbors - 1))) * np.sum(
        np.abs(
            (embedded_ranks[:, :n_neighbors] - original_ranks[:, :n_neighbors]) > 0
        )
    )
    return continuity_score

cont = continuity(data_course_CR, data_tsne, n_neighbors=10)
print(f"Continuity: {cont:.4f}")


# __________________________________________________

rootTSNE = tk.Tk()
rootTSNE.title("Graphique t-SNE")
sns.set_style("white")

# Création de la figure et des axes Matplotlib
figTSNE, axTSNE = plt.subplots(figsize=(10, 6))

# Création d'une palette de couleurs basée sur Color Brewer ("Dark2")
palette = plt.get_cmap("tab20")

# Associer une couleur à chaque nom distinct
couleurs = dict(zip(noms_distincts, palette(range(N))))

# Affichage des points avec des couleurs spécifiques par personne et une taille réduite
for nom in noms_distincts:
    subset = data_tsne_df[data_tsne_df["Nom"] == nom]
    axTSNE.scatter(
        subset["Dim1"], subset["Dim2"],
        label=nom,
        color=couleurs[nom],
        s=5)

# Ajouter des labels, un titre et une légende
axTSNE.set_xlabel("Dimension 1")
axTSNE.set_ylabel("Dimension 2")
axTSNE.set_title("Projection t-SNE (2D)")

# Ajouter une légende dynamique
axTSNE.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Légende")

figTSNE.savefig("/Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/resultats t-SNE/Plot-tSNE/Projection t-SNE (2D).png", format='png', dpi=300)  # `dpi=300` pour une haute qualité


# Intégrer le graphique Matplotlib dans la fenêtre Tkinter
canvasTSNE = FigureCanvasTkAgg(figTSNE, master=rootTSNE)
canvasTSNE.draw()
canvasTSNE.get_tk_widget().pack(expand=True, fill="both")

# Ajouter une barre d'outils de navigation (avec zoom et pan)
toolbarTSNE = NavigationToolbar2Tk(canvasTSNE, rootTSNE)
toolbarTSNE.update()
toolbarTSNE.pack(side=tk.BOTTOM, fill=tk.X)

# Lancer l'application Tkinter
rootTSNE.mainloop()


#__________________________________________________

rootCOURSE = tk.Tk()
rootCOURSE.title("Graphique t-SNE")
sns.set_style("white")

# Nombre de graphiques

rows = int(np.ceil(N / 4))  # Nombre de lignes
cols = min(4, N)  # Nombre de colonnes, max 3 par ligne

figCOURSE, axes = plt.subplots(rows, cols, figsize=(10, 10))

# Si la grille est 1D, convertir `axes` en une liste pour itération
axes = axes.flatten() if N > 1 else [axes]

# Ajouter des graphiques aux sous-graphiques
for i in range(N):
    subset = data_tsne_df[data_tsne_df["Nom"] == noms_distincts[i]]
    axes[i].scatter(subset["Dim1"], subset["Dim2"], label=noms_distincts[i], color=couleurs[noms_distincts[i]], s=5)
    axes[i].set_title(f"Graphique {noms_distincts[i]}")
    axes[i].set_xlabel("dim1")
    axes[i].set_ylabel("dim2")

# Masquer les sous-graphiques inutilisés (si N ne remplit pas toute la grille)
for i in range(N, len(axes)):
    axes[i].axis('off')  # Cacher le sous-graphe

# Ajuster l'espace entre les sous-graphiques
plt.tight_layout()


figCOURSE.savefig("/Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/resultats t-SNE/Plot-tSNE/Graphique t-SNE par pers.png", format='png', dpi=300)  # `dpi=300` pour une haute qualité


# Intégrer le graphique Matplotlib dans la fenêtre Tkinter
canvasCOURSE = FigureCanvasTkAgg(figCOURSE, master=rootCOURSE)
canvasCOURSE.draw()
canvasCOURSE.get_tk_widget().pack(expand=True, fill="both")

# Ajouter une barre d'outils de navigation (avec zoom et pan)
toolbarCOURSE = NavigationToolbar2Tk(canvasCOURSE, rootCOURSE)
toolbarCOURSE.update()
toolbarCOURSE.pack(side=tk.BOTTOM, fill=tk.X)

# Lancer l'application Tkinter
rootCOURSE.mainloop()

#__________________________________________________

root3D = tk.Tk()
root3D.title("Graphique t-SNE")
sns.set_style("white")

# Créer une figure
fig3D = plt.figure()
ax3D = fig3D.add_subplot(111, projection='3d')

# Création d'une palette de couleurs basée sur Color Brewer ("Dark2")
palette3D = plt.get_cmap("Dark2")

# Créer une palette de couleurs pour chaque nom distinct
couleurs2 = palette3D(np.linspace(0, 1, len(noms_distincts)))
colors = [
    'red', 'blue', 'green', 'orange', 'purple', 'cyan', 
    'magenta', 'yellow', 'brown', 'pink', 'lime', 'teal', 
    'navy', 'olive', 'maroon', 'gold', 'gray', 'black', 
    'violet', 'indigo'
]

for i, nom in enumerate(noms_distincts):
    subset = data_tsne_df[data_tsne_df["Nom"] == nom]
    ax3D.scatter(
        subset["Dim1"], subset["Dim2"], subset["Dim3"],
        label=nom, 
        color=[couleurs2[i]],  # Passer la couleur en liste pour chaque 'nom'
        s=5)

# Ajouter des labels, un titre et une légende
ax3D.set_xlabel("Dimension 1")
ax3D.set_ylabel("Dimension 2")
ax3D.set_zlabel("Dimension 3")
ax3D.set_title("Projection des données")

# Ajouter une légende dynamique
ax3D.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Légende")

fig3D.savefig("/Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/resultats t-SNE/Plot-tSNE/Projection des données 3D.png", format='png', dpi=300)  # `dpi=300` pour une haute qualité

# Intégrer le graphique Matplotlib dans la fenêtre Tkinter
canvas3D = FigureCanvasTkAgg(fig3D, master=root3D)
canvas3D.draw()
canvas3D.get_tk_widget().pack(expand=True, fill="both")

# Ajouter une barre d'outils de navigation (avec zoom et pan)
toolbar3D = NavigationToolbar2Tk(canvas3D, root3D)
toolbar3D.update()
toolbar3D.pack(side=tk.BOTTOM, fill=tk.X)

# Lancer l'application Tkinter
root3D.mainloop()


data_complet.to_csv('/Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/resultats t-SNE/Data_complet.csv', index=False)
data_course.to_csv('/Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/resultats t-SNE/Data_complet_av_tTSNE.csv', index=False)
data_course_CR.to_csv('/Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/resultats t-SNE/Data_complet_av_tSNE_apCR_OL.csv', index=False)
data_course_CRnom.to_csv('/Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/resultats t-SNE/Data_complet_av_tSNE_apCR_OLnom.csv', index=False)
data_tsne_df.to_csv('/Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/resultats t-SNE/Data_complet_ap_tSNE.csv', index=False)
data_course_CR_wOLnom.to_csv('/Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/resultats t-SNE/Data_complet_av_tSNE_CRnom.csv', index=False)


#------------------------------------------------------------------------------
# Algo Clustering
#------------------------------------------------------------------------------


# =============================================================================
# Variables
# =============================================================================

nb_clusters = 3
cluster_data = []  # DataFrame pour chaque point avec son cluster et ses dimensions
clusters = [[] for _ in range(nb_clusters)]  # Initialiser une liste vide pour contenir les clusters
data_cluster_tot = [[] for _ in range(4)]
data_new = []

Cluster1 = pd.DataFrame()
Cluster1List = []
Cluster2 = pd.DataFrame()
Cluster2List = []
Cluster3 = pd.DataFrame()
Cluster3List = []
Cluster123 = pd.DataFrame()
Cluster4 = pd.DataFrame()
Cluster4List = []

# =============================================================================
# Intro de fenetre
# =============================================================================

data = pd.read_csv('/Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/resultats t-SNE/Data_complet_ap_tSNE.csv')

noms_distincts = list(data["Nom"].unique())
N = len(noms_distincts)

# Sélectionner les dimensions pertinentes pour le clustering (Dim1, Dim2, Dim3)
data_3dim = data[["Dim1", "Dim2", "Dim3"]].values

rootClusterSimpleLinkage = tk.Tk()
rootClusterSimpleLinkage.title("Graphique t-SNE Cluster Simple Linkage")
sns.set_style("white")

figClusterSimpleLinkage = plt.figure()
axClusterSimpleLinkage = figClusterSimpleLinkage.add_subplot(111, projection='3d')

colors = [
    'red', 'blue', 'green', 'orange', 'purple', 'cyan', 
    'magenta', 'yellow', 'brown', 'pink', 'lime', 'teal', 
    'navy', 'olive', 'maroon', 'gold', 'gray', 'black', 
    'violet', 'indigo'
]

# =============================================================================
# Clustering Simple Linkage
# =============================================================================

# Effectuer le clustering avec la méthode "single linkage"
linkage_matrix = linkage(data_3dim, method='single')
labels = fcluster(linkage_matrix, nb_clusters, criterion='maxclust')
data['Cluster'] = labels

# Ajouter les informations du cluster pour chaque point
for idx, row in data.iterrows():
    cluster_data.append([row['Cluster'], row['Dim1'], row['Dim2'], row['Dim3'], row['Nom']])

# Créer un DataFrame avec les résultats
df_clusters = pd.DataFrame(cluster_data, columns=["Cluster", "Dim1", "Dim2", "Dim3", "Nom"])

# Remplir chaque liste de cluster avec les points associés
for idx, row in df_clusters.iterrows():
    cluster_index = int(row['Cluster']) - 1  # Indice du cluster (1 à nb_clusters)
    clusters[cluster_index].append([row['Dim1'], row['Dim2'], row['Dim3'], row["Nom"]])  # Ajouter les coordonnées du point

for idx, row in df_clusters.iterrows():
    new_row = {'Dim1': row['Dim1'], 'Dim2': row['Dim2'], 'Dim3': row['Dim3'], 'Nom': row["Nom"]}
    new_df = pd.DataFrame([new_row])
    if row['Cluster']==2:
        Cluster4 = pd.concat([Cluster4, new_df], ignore_index=True)
        Cluster4List = Cluster4.to_numpy()
    else:
        Cluster123 = pd.concat([Cluster123, new_df], ignore_index=True)

kmeans = KMeans(n_clusters=nb_clusters, random_state=42)
Cluster123['Cluster123'] = kmeans.fit_predict(Cluster123.iloc[:, 0:3])

# affinity = AffinityPropagation(damping=0.9).fit(Cluster123.iloc[:, 0:3])
# Cluster123['Cluster123'] = affinity.labels_

# birch = Birch(n_clusters=None).fit(Cluster123.iloc[:, 0:3])
# Cluster123['Cluster123'] = birch.labels_

# minibatch_kmeans = MiniBatchKMeans(n_clusters=nb_clusters, random_state=42, batch_size=100)
# Cluster123['Cluster123'] = minibatch_kmeans.fit_predict(Cluster123.iloc[:, 0:3])

# gmm = GaussianMixture(n_components=nb_clusters, random_state=42) 
# Cluster123['Cluster123'] = gmm.fit_predict(Cluster123.iloc[:, 0:3]) 

for idx, row in Cluster123.iterrows():
    new_row = {'Dim1': row['Dim1'], 'Dim2': row['Dim2'], 'Dim3': row['Dim3'], 'Nom': row["Nom"]}
    new_df = pd.DataFrame([new_row])
    if row['Cluster123']==0:
        Cluster1 = pd.concat([Cluster1, new_df], ignore_index=True)
        Cluster1List = Cluster1.to_numpy()
    elif row['Cluster123']==1:
        Cluster2 = pd.concat([Cluster2, new_df], ignore_index=True)
        Cluster2List = Cluster2.to_numpy()
    elif row['Cluster123']==2:
        Cluster3 = pd.concat([Cluster3, new_df], ignore_index=True)
        Cluster3List = Cluster3.to_numpy()


i = 0
# Parcourez chaque ligne et ajoutez les coordonnées au cluster correspondant
for list in [Cluster1, Cluster2, Cluster3, Cluster4]:
    for idx, row in list.iterrows():
        cluster_index = i  # Identifie le cluster
        p = [row['Dim1'], row['Dim2'], row['Dim3'], row["Nom"]]  # Coordonnées du point
        data_cluster_tot[cluster_index].append(p)
    i = i + 1
    

# Parcourir chaque cluster et ses points
for i, cluster in enumerate(data_cluster_tot):
    for points in cluster:
        data_new.append([points[0], points[1], points[2], i+1, points[3]])  # Ajout direct des dimensions et du cluster

# Créer le DataFrame
Data_new = pd.DataFrame(data_new, columns=['Dim1', 'Dim2', 'Dim3', 'Cluster', 'Nom'])

    
# =============================================================================
# Visualisation (clusters)
# =============================================================================
data_cluster_tot2 = data_cluster_tot.copy()
for liste in data_cluster_tot2:
    for liste1 in liste:
        liste1.pop(3)

# Couleurs pour chaque cluster
colors = ['red', 'green', 'blue', 'purple']

# Parcours de chaque cluster
for i, cluster in enumerate(data_cluster_tot2):
    # Conversion de la liste de points en un tableau NumPy pour faciliter le plot
    cluster_array = np.array(cluster)

    # Plot des points du cluster
    axClusterSimpleLinkage.scatter(cluster_array[:, 0], cluster_array[:, 1], cluster_array[:, 2], color=colors[i], label=f'Cluster {i+1}', s=5)

    # Calcul du centroïde
    centroid = np.mean(cluster_array, axis=0)
    axClusterSimpleLinkage.scatter(centroid[0], centroid[1], centroid[2], color='black', marker='o', s=50)

# Ajouter des labels, un titre et une légende
axClusterSimpleLinkage.set_xlabel("Dimension 1")
axClusterSimpleLinkage.set_ylabel("Dimension 2")
axClusterSimpleLinkage.set_zlabel("Dimension 3")
axClusterSimpleLinkage.set_title("Plan factoriel")

# Ajouter une légende dynamique
axClusterSimpleLinkage.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Légende")

figClusterSimpleLinkage.savefig("/Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/resultats t-SNE/Plot-tSNE/Projection cluster simple linkage 3D.png", format='png', dpi=300)  # `dpi=300` pour une haute qualité

# Intégrer le graphique Matplotlib dans la fenêtre Tkinter
canvasCluster = FigureCanvasTkAgg(figClusterSimpleLinkage, master=rootClusterSimpleLinkage)
canvasCluster.draw()
canvasCluster.get_tk_widget().pack(expand=True, fill="both")

# Ajouter une barre d'outils de navigation (avec zoom et pan)
toolbarCluster = NavigationToolbar2Tk(canvasCluster, rootClusterSimpleLinkage)
toolbarCluster.update()
toolbarCluster.pack(side=tk.BOTTOM, fill=tk.X)

# Lancer l'application Tkinter
rootClusterSimpleLinkage.mainloop()


# =============================================================================
# Analyser le clustering - Barres empilées
# =============================================================================


# Grouper par 'Nom' et 'Cluster' et compter les points dans chaque groupe
cluster_counts = Data_new.groupby(['Nom', 'Cluster']).size().reset_index(name='Point_Count')
pivot_table = cluster_counts.pivot(index='Nom', columns='Cluster', values='Point_Count').fillna(0)
pivot_table["total"] = pivot_table.sum(axis=1)
pivot_table.iloc[:, 0:9] = pivot_table.iloc[:, 0:9].div(pivot_table['total'], axis=0)

pivot_table.to_csv('/Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/pivot_table.csv', index=False)

# Supprimer la colonne "total" (non pertinente pour les barres empilées)
data_without_total = pivot_table.drop(columns=['total'])
# Ajouter une ligne de somme des colonnes
data_with_total = data_without_total.copy()
data_with_total.loc['Total'] = data_without_total.sum()


# =============================================================================
# Analyser le clustering - Headmap des données
# =============================================================================


# Créer un masque pour exclure la ligne "Total" de la coloration
mask = np.zeros_like(data_with_total, dtype=bool)
mask[-1, :] = True

# Définir la figure et les axes
fig, ax = plt.subplots(figsize=(12, 6))

# Créer la heatmap avec le masque
sns.heatmap(
    data_with_total,
    annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, 
    mask=mask, cbar_kws={'label': 'Valeurs'}, ax=ax)

# Ajouter les annotations en rouge pour la ligne "Total"
for col_idx, value in enumerate(data_with_total.iloc[-1]):
    ax.text(
        col_idx + 0.5, data_with_total.shape[0] - 0.5,  # Position centrale dans la cellule
        f"{value:.2f}", color="red", fontsize=10, ha="center", va="center"
    )

plt.title("Heatmap des données")
plt.xlabel("Colonnes")
plt.ylabel("Noms")
plt.tight_layout()
plt.savefig("/Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/resultats t-SNE/Plot-tSNE/Analyser le clustering - Headmap des données.png", format='png', dpi=300)  # `dpi=300` pour une haute qualité
plt.show()


# =============================================================================
# Distance 
# =============================================================================

# Extraire les coordonnées (colonnes 1 à 4)
coordinates = data_without_total.iloc[0:N,:].to_numpy()

# Calculer les distances euclidiennes par paires
distances = squareform(pdist(coordinates, metric='euclidean'))

# Convertir en DataFrame pour une meilleure lisibilité
distance_df = pd.DataFrame(
    distances,
    columns=[f'Person_{i}' for i in range(len(coordinates))],
    index=[f'Person_{i}' for i in range(len(coordinates))]
)

# Appliquer l'algorithme K-means
n_clusters2 = 4  # Nombre de clusters souhaités
kmeans = KMeans(n_clusters=n_clusters2, random_state=42)
kmeans.fit(coordinates)



# Ajouter les labels des clusters au DataFrame
data_without_total['Cluster'] = kmeans.labels_


root = tk.Tk()
root.title("Graphique")
sns.set_style("white")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Visualisation des clusters (si possible, avec 2 dimensions principales)
scatter = ax.scatter(data_without_total.iloc[0:N, 0], data_without_total.iloc[0:N, 1], data_without_total.iloc[0:N, 2], c=kmeans.labels_, cmap='viridis')
# Ajouter les étiquettes des axes
ax.set_title('K-means Clustering (3D)')
ax.set_xlabel('Component 1')
ax.set_ylabel('Component 2')
ax.set_zlabel('Component 3')

# # Ajouter une barre de couleurs
# fig.colorbar(scatter, ax=ax, label='Cluster')
# Enregistrer les noms des lignes (index) dans un fichier texte
fichier_sortie = os.path.join('/Users/camilleauvity/Desktop/ECOLE/Projet-Industriel', "noms_lignes.txt")
with open(fichier_sortie, "w") as f:
    for index in data_without_total.index:
        f.write(f"{index}\n")
        
data_without_total.to_csv('/Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/data_without_total1.csv', index=False)

# Ajouter les noms des personnes à côté des points
for i, name in enumerate(data_without_total.index.to_list()):
    ax.text(coordinates[i, 0], coordinates[i, 1], coordinates[i, 2], name, fontsize=8)

plt.savefig("/Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/RESULTATS//Kmeans_Clustering.png", format='png', dpi=300)

plt.show()

# Intégrer le graphique Matplotlib dans la fenêtre Tkinter
canvasCoord = FigureCanvasTkAgg(fig, master=root)
canvasCoord.draw()
canvasCoord.get_tk_widget().pack(expand=True, fill="both")

# Ajouter une barre d'outils de navigation (avec zoom et pan)
toolbarCluster = NavigationToolbar2Tk(canvasCoord, root)
toolbarCluster.update()
toolbarCluster.pack(side=tk.BOTTOM, fill=tk.X)

# Lancer l'application Tkinter
root.mainloop()

#------------------------------------------------------------------------------
# Algo proximité
#------------------------------------------------------------------------------


def creer_tableau_proximite(data):
    """
    Crée un tableau associant chaque personne à sa personne la plus proche dans le même cluster.
    
    Args:
        data (pd.DataFrame): Le dataframe contenant les données.
        
    Returns:
        pd.DataFrame: Un dataframe avec les colonnes 'Personne' et 'Personne la plus proche'.
    """
    resultats = []

    # Boucle sur chaque personne
    for personne in data.index:
        # Récupérer le cluster de la personne
        cluster_cible = data.loc[personne, 'Cluster']
        
        # Filtrer les données pour les personnes dans le même cluster
        data_meme_cluster = data[data['Cluster'] == cluster_cible]
        
        # Supprimer la colonne Cluster pour le calcul et s'assurer d'exclure la personne actuelle
        data_sans_cluster = data_meme_cluster.drop(columns=['Cluster'])
        distances = data_sans_cluster.apply(
            lambda row: np.linalg.norm(row - data_sans_cluster.loc[personne]), axis=1
        )
        
        # Supprimer la distance à soi-même
        distances = distances.drop(personne)
        
        # Vérifier si distances n'est pas vide
        if not distances.empty:
            # Trouver la personne la plus proche
            personne_proche = distances.idxmin()
        else:
            personne_proche = None  # Aucun voisin dans le même cluster
        
        # Ajouter au tableau des résultats
        resultats.append({'Personne': personne, 'Personne la plus proche': personne_proche})
    
    # Convertir en DataFrame
    return pd.DataFrame(resultats)

# Exemple d'utilisation
tableau_proximite = creer_tableau_proximite(data_without_total)


def ajouter_nom_fichier_proche_10km(data):
    """
    Ajoute une colonne contenant le nom de fichier de la course dont la distance est la plus proche de 10 km.
    
    Args:
        data (pd.DataFrame): Le dataframe contenant les informations des courses.
    
    Returns:
        pd.DataFrame: Un dataframe avec une colonne supplémentaire 'Nom fichier proche 10km'.
    """
    # Créer une liste pour stocker les résultats
    resultats = []
    
    # Grouper les données par personne (colonne 'Nom')
    groupes = data.groupby("Nom")
    
    for nom, groupe in groupes:
        # Calculer la différence absolue entre la distance et 10
        groupe['Diff_10km'] = (groupe['Distance'] - 10).abs()
        
        # Identifier la ligne avec la distance la plus proche de 10 km
        ligne_proche_10km = groupe.loc[groupe['Diff_10km'].idxmin()]
        
        # Extraire uniquement la partie droite du slash dans 'Nom du fichier'
        nom_fichier_proche_10km = ligne_proche_10km['Nom du fichier'].split("/")[-1]
        
        # Ajouter le résultat pour la personne
        resultats.append({
            "Nom": nom,
            "Nom fichier proche 10km": nom_fichier_proche_10km
        })
    
    # Transformer les résultats en DataFrame
    
    df_resultats = pd.DataFrame(resultats)
    return df_resultats



filtered_data = data_cleaned_complet[data_cleaned_complet['Nom du fichier'].str.endswith('.gpx', na=False)]
Nom_gpx_personne = ajouter_nom_fichier_proche_10km(filtered_data)

j1 = 0
for i in tableau_proximite['Personne la plus proche']:
    if i != None:
        # Filtrer pour obtenir la ligne correspondante dans Nom_gpx_personne
        fichier = Nom_gpx_personne.loc[Nom_gpx_personne['Nom'] == i, 'Nom fichier proche 10km']
        
        # Vérifier que le fichier existe avant d'assigner
        if not fichier.empty:
            tableau_proximite.loc[j1, "GPX"] = fichier.values[0]
        else:
            tableau_proximite.loc[j1, "GPX"] = None  # Si aucune correspondance n'est trouvée
        
        j1 += 1

tableau_proximite.to_csv("/Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/Evo-Pace-Selon-Distance/tableau_proximite.csv", index=False)


#------------------------------------------------------------------------------
# Algo Clustering
#------------------------------------------------------------------------------


warnings.filterwarnings("ignore")


# Attempting again with the newly uploaded file
data1 = pd.read_csv("/Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/DataBaseTotal.csv")

noms_distincts = data1["Nom"].unique()
N = len(noms_distincts)

# Process the data as before
data_complet = data1.dropna()
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

data_cleaned_complet.to_csv("/Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/data_cleaned_complet.csv", index=False)


# Reorganize the cleaned dataset
data_course_CR = data_cleaned_loop.drop(columns=["nom"])
data_course_CR["Nom"] = data_cleaned_complet["Nom"]
data_course_CR["ID de l'activité"] = data_cleaned_complet["ID de l'activité"]
data_course_CR["Date de l'activité"] = data_cleaned_complet["Date de l'activité"]

# Calculate average speed (km/h) and format the date
data_course_CR['Vitesse Moyenne (min/km)'] =  data_cleaned_complet['Temps en minutes'] / (data_course_CR['Distance ajustée selon la pente'] / 1000)

# Initialiser une liste pour stocker les résultats
regression_results = []


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


#------------------------------------------------------------------------------
# evolution Pace
#------------------------------------------------------------------------------


locale.setlocale(locale.LC_TIME, 'fr_FR.UTF-8')
warnings.filterwarnings("ignore")

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

french_months = {
    "janvier": "Jan",
    "février": "Feb",
    "mars": "Mar",
    "avril": "Apr",
    "mai": "May",
    "juin": "Jun",
    "juillet": "Jul",
    "août": "Aug",
    "septembre": "Sep",
    "octobre": "Oct",
    "novembre": "Nov",
    "décembre": "Dec"
}

# Normaliser les dates
def normaliser_mois(date):
    for mois_abrege, mois_complet in mois_francais.items():
        date = date.replace(mois_abrege, mois_complet)
    return date

def inv_mois(date):
    for mois_fr, mois_en in french_months.items():
        date = date.replace(mois_en, mois_fr)
    return date

# Fonction pour ajouter une prédiction des performances futures
def predict_future_performance(dates, performances, future_days, degree=2):
   
    # Conversion des dates en nombres de jours depuis la première date
    date_numeric = [(date - dates.min()).days for date in dates]

    # Ajustement d'un modèle polynomial
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(np.array(date_numeric).reshape(-1, 1))
    model = LinearRegression()
    model.fit(X_poly, performances)

    # Générer des dates futures
    future_date_numeric = [date_numeric[-1] + i for i in range(1, future_days + 1)]
    future_dates = [dates.max() + pd.Timedelta(days=i) for i in range(1, future_days + 1)]

    # Prédictions
    future_X_poly = poly.transform(np.array(future_date_numeric).reshape(-1, 1))
    future_predictions = model.predict(future_X_poly)

    return future_dates, future_predictions

# Attempting again with the newly uploaded file
data2 = pd.read_csv("/Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/DataBaseTotal.csv")


# Charger le tableau des coefficients
regression_coefficients = pd.read_csv("/Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/Evo-Pace-Selon-Distance/Regression_Coefficients.csv")


noms_distincts = data2["Nom"].unique()
N = len(noms_distincts)

# Process the data as before
data_complet = data2.dropna()
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
    
    lof = LocalOutlierFactor(n_neighbors=10, algorithm='auto', metric='euclidean', contamination=0.1)
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
data_course_CR['Vitesse Moyenne (km/h)'] =  data_cleaned_complet['Temps en minutes'] / (data_course_CR['Distance ajustée selon la pente'] / 1000)

# Extraction de la date pour les formats avec " à "
data_course_CR['Cleaned Date'] = data_course_CR['Date de l\'activité'].str.split(' à ').str[0]
  
data_course_CR['Normalized Date'] = data_course_CR['Cleaned Date'].apply(normaliser_mois)

# Conversion des dates normalisées en format datetime
data_course_CR['Parsed Date'] = pd.to_datetime(data_course_CR['Normalized Date'], format='%d %B %Y', errors='coerce', dayfirst=True)

# Formater les dates dans le format souhaité (dd/mm/yyyy)
data_course_CR['Formatted Date'] = data_course_CR['Parsed Date'].dt.strftime('%d/%m/%Y')

# Boucle sur toutes les lignes
for k in range(data_course_CR.shape[0]):
    date_test = data_course_CR['Formatted Date'].iloc[k]
    
    # Vérifier si la date est manquante
    if pd.isna(date_test):
        # Extraction pour les formats avec ", "
        mot = data_course_CR['Date de l\'activité'].iloc[k]
        
        if len(mot.split(',')) == 2:
            divmot = mot.split()
            # Conversion de la chaîne en un objet datetime
            date_obj = mot.replace(divmot[1],inv_mois(divmot[1]))
            # Conversion en objet datetime
            date_obj = datetime.strptime(date_obj, '%d %B %Y, %H:%M:%S')

            # Formatage de l'objet datetime
            formatted_date = date_obj.strftime('%d/%m/%Y')

            data_course_CR.at[k, 'Parsed Date'] = formatted_date

        else:
            # Séparer la date avec ", "
            
            mot1 = mot.split(', ')[0]  # Mois et jour (ex. "May 28")
            mot11 = mot1.split(' ')[0] 
            mot12 = mot1.split(' ')[1] 
    
            mot2 = mot.split(', ')[1]  # Année (ex. "2024")
            
            # Créer la nouvelle colonne 'Cleaned Date'
            data_course_CR.at[k, 'Cleaned Date'] = mot12 + " " + mot11 + " " + mot2
            data_course_CR.at[k, 'Cleaned Date'] = inv_mois(data_course_CR.at[k, 'Cleaned Date'])
    
            norm_date = data_course_CR.at[k, 'Cleaned Date']
            data_course_CR.at[k, 'Normalized Date'] = norm_date
            pers_date = pd.to_datetime(norm_date, format='%d %B %Y', errors='coerce', dayfirst=True)
            data_course_CR.at[k, 'Parsed Date'] = pers_date
            form_date = pers_date.strftime('%d/%m/%Y')


# Formater les dates dans le format souhaité (dd/mm/yyyy)
data_course_CR['Formatted Date'] = data_course_CR['Parsed Date'].dt.strftime('%d/%m/%Y')


# Tri chronologique par Formatted Date pour chaque personne
person_data_cat = pd.DataFrame()
person_tables = {}
for name in data_course_CR["Nom"].unique():
    person_data = data_course_CR[data_course_CR["Nom"] == name][['Parsed Date', 'Distance ajustée selon la pente', "Vitesse Moyenne (km/h)"]]
    person_data["Nom"] = name
    person_data["Temps (min)"] = data_course_CR["Temps en minutes"]
    person_data["Distance"] = data_course_CR["Distance"]
    person_tables[name] = person_data.sort_values(by='Parsed Date',axis = 0, ascending=True)
    person_data_cat = pd.concat([person_data_cat,person_data],ignore_index=True)
    
person_data_cat.to_csv('/Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/person_data_cat.csv', index=False)


# Fonction de moyenne glissante
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


# Paramètres ARIMA
order = (2, 1, 2)  # Paramètres p, d, q (ajuster si nécessaire)

# Tracer les graphiques pour chaque personne avec les données simulées
for name in person_data_cat['Nom'].unique():
    
    # Filtrer les données pour chaque personne
    person_subset = person_data_cat[person_data_cat['Nom'] == name].sort_values(by='Parsed Date')
    
    npjourpred = int(len(person_subset['Parsed Date']) * 0.5)
    
    # Vérifier si les coefficients pour cette personne existent
    coefficients_row = regression_coefficients[regression_coefficients['Nom'] == name]

    # Extraire les coefficients
    intercept = coefficients_row['Intercept'].values[0]
    coefficients = [
        coefficients_row[f"Coefficient_x^{i}"].values[0] for i in range(1, len(coefficients_row.columns) - 2)
    ]
    
    # Évaluer la fonction polynomiale à chaque distance
    distances = person_subset['Distance'].values
    polynomial_values = np.polyval([*coefficients[::-1], intercept], distances)
    
    # Ajuster la vitesse moyenne en multipliant par la valeur de la fonction polynomiale
    person_subset['Vitesse Moyenne Modifiée (km/h)'] = (
        person_subset['Vitesse Moyenne (km/h)'] * polynomial_values
    )
    
    # Calculer la moyenne et l'écart-type de la vitesse modifiée
    mean_speed = person_subset['Vitesse Moyenne Modifiée (km/h)'].mean()
    std_dev_speed = person_subset['Vitesse Moyenne Modifiée (km/h)'].std()
    
    # Calculer les limites de l'intervalle
    lower_bound = mean_speed - 2.96 * std_dev_speed
    upper_bound = mean_speed + 2.96 * std_dev_speed
    
    # Filtrer les points à l'intérieur de l'intervalle
    filtered_subset = person_subset[
        (person_subset['Vitesse Moyenne Modifiée (km/h)'] >= lower_bound) &
        (person_subset['Vitesse Moyenne Modifiée (km/h)'] <= upper_bound)
    ]
    
    # Appliquer la moyenne glissante
    window_size = 5  # Définir la taille de la fenêtre
    y_smooth = moving_average(filtered_subset['Vitesse Moyenne Modifiée (km/h)'].values, window_size)
    
    x_smooth = filtered_subset['Parsed Date'].iloc[(window_size - 1) // 2 : -(window_size - 1) // 2].values
    
    # ---- Modèle ARIMA ----
    # Extraire les données nécessaires
    speeds = person_subset['Vitesse Moyenne Modifiée (km/h)'].values
    
    # Ajuster le modèle ARIMA
    model = SARIMAX(speeds, order=order, enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit(disp=False)

    # Générer les prédictions futures (par exemple, 7 jours)
    future_steps = 12
    future_predictions = results.get_forecast(steps=future_steps)
    forecast_values = future_predictions.predicted_mean
    confidence_intervals = future_predictions.conf_int()
    
    # Générer des dates futures pour les prédictions
    future_dates = [
        person_subset['Parsed Date'].iloc[-1] + pd.Timedelta(days=i*4) for i in range(1, future_steps + 1)
    ]

    # Tracer le graphique
    plt.figure(figsize=(10, 6))
    
    # Ajouter les prédictions futures au graphique
    plt.plot(future_dates, forecast_values, label="Prédictions futures (ARIMA)", color='green', linestyle='--')
    plt.fill_between(
        future_dates,
        forecast_values*0.95,  # Utilisation de l'indexation numpy
        forecast_values*1.05,
        color='green',
        alpha=0.2,
        label="Intervalle de confiance (95%)"
    )
    
    # Tracer les données lissées et filtrées
    plt.plot(x_smooth, y_smooth, label=f'Moyenne glissante (window={window_size})', color='red', linewidth=2)
    plt.plot(filtered_subset['Parsed Date'], filtered_subset['Vitesse Moyenne Modifiée (km/h)'], 
             marker='o', label="Vitesse Moyenne Modifiée (filtrée)")
    plt.fill_between(filtered_subset['Parsed Date'], 
                     mean_speed - std_dev_speed, 
                     mean_speed + std_dev_speed, 
                     color='orange', alpha=0.2, label="± Écart-type")
    
    plt.axhline(mean_speed, color='red', linestyle='--', label=f"Moyenne ({mean_speed:.2f})")

    # Ajouter le titre et les légendes
    plt.title(f"Évolution de la vitesse moyenne modifiée pour {name} avec prédictions futures (ARIMA)")
    plt.xlabel("Date")
    plt.ylabel("Vitesse Moyenne Modifiée (km/h)")
    plt.grid(True)
    plt.legend()
    
    # Configuration de l'axe X pour respecter l'espacement temporel
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gca().xaxis.set_major_formatter(DateFormatter("%d/%m/%Y"))
    plt.gcf().autofmt_xdate()
    
    # Sauvegarder le graphique
    plt.savefig(
        f"/Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/Evo-Pace-Selon-Distance/Evolution_Pace_Modifiée_{name}_ARIMA.png",
        format='png',
        dpi=300
    )
    plt.savefig(f"/Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/RESULTATS/resultats_{name}/Evolution_Pace_Modifiée_{name}_ARIMA.png", format='png', dpi=300)

    plt.show()
    
    
    

from sklearn.linear_model import LinearRegression

# Définir les distances pour lesquelles nous voulons suivre les records
target_distances = [3, 5, 10, 15, 21]  # Distances en km

# Tracer les graphiques pour chaque personne
for name in person_data_cat['Nom'].unique():
    # Filtrer les données pour chaque personne
    person_subset = person_data_cat[person_data_cat['Nom'] == name].sort_values(by='Parsed Date')
    
    # Initialiser un dictionnaire pour stocker les records par distance
    records = {dist: [] for dist in target_distances}
    dates = {dist: [] for dist in target_distances}
    
    # Parcourir chaque ligne et mettre à jour les records
    for _, row in person_subset.iterrows():
        distance = row['Distance']
        time = row['Temps (min)']  # Supposons que cette colonne contient le temps en minutes
        date = row['Parsed Date']
        
        for target in target_distances:
            if distance >= target:
                # Calculer le temps pour atteindre la distance cible (proportionnel)
                target_time = (target / distance) * time
                
                # Si c'est le premier record ou une amélioration, on le sauvegarde
                if not records[target] or target_time < records[target][-1]:
                    records[target].append(target_time)
                    dates[target].append(date)
                else:
                    # Si pas de nouveau record, conserver le précédent
                    records[target].append(records[target][-1])
                    dates[target].append(date)
    
    # Tracer les records pour chaque distance
    plt.figure(figsize=(12, 8))
    for target in target_distances:
        # Convertir les dates en format numérique pour la régression
        dates_numeric = mdates.date2num(dates[target])  # Convertir les dates pour le modèle
        records_array = np.array(records[target])

        # Vérifier qu'il y a suffisamment de données pour une régression
        if len(dates_numeric) < 3:
            continue

        # Appliquer une régression linéaire
        linear_model = LinearRegression()
        X = dates_numeric.reshape(-1, 1)  # Reshape pour le modèle
        y = records_array

        # Entraîner le modèle
        linear_model.fit(X, y)

        # Faire des prédictions sur une plage de dates futures
        future_dates = np.linspace(X[-1], X[-1] + 30, 100).reshape(-1, 1)  # 30 jours dans le futur
        y_pred = linear_model.predict(future_dates)

        # Calculer l'incertitude (± 5% de la prédiction)
        uncertainty_margin = 0.05 * y_pred  # 5% d'incertitude
        lower_bound = y_pred #np.maximum(y_pred - uncertainty_margin, records[target][-1])  # Pas en-dessous de la dernière valeur
        upper_bound = np.minimum(y_pred + uncertainty_margin, records[target][-1])  # Pas au-dessus de la dernière valeur

        # Tracer les données observées
        plt.plot(dates[target], records[target], label=f"Record {target} km")

        # Tracer les prédictions futures
        future_dates_plot = mdates.num2date(future_dates.flatten())  # Convertir en format datetime
        plt.plot(future_dates_plot, y_pred, label=f"Prédiction future {target} km", linestyle='--')

        # Ajouter la plage d'incertitude
        plt.fill_between(
            future_dates_plot,
            lower_bound,
            upper_bound,
            color='gray',
            alpha=0.2,
            label=f"Incertitude {target} km"
        )

    # Ajouter des titres et légendes
    plt.title(f"Évolution des records pour {name}")
    plt.xlabel("Date")
    plt.ylabel("Temps (min)")
    plt.grid(True)
    plt.legend()
    
    # Configuration de l'axe X pour les dates
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gca().xaxis.set_major_formatter(DateFormatter("%d/%m/%Y"))
    plt.gcf().autofmt_xdate()
    plt.savefig(f"/Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/Evo-Pace-Selon-Distance/Evolution_Records_Linear_{name}.png", format='png', dpi=300)
    plt.savefig(f"/Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/RESULTATS/resultats_{name}/Evolution_Records_Linear_{name}.png", format='png', dpi=300)

    plt.show()

