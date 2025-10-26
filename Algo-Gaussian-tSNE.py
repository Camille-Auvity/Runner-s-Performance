
import os
import math
import pandas 
import random
import seaborn
import numpy as np
import tkinter as tk
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


# =============================================================================
# Intro de fenetre
# =============================================================================

data = pandas.read_csv('/Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/resultats t-SNE/Data_complet_ap_tSNE.csv')

# Sélectionner les dimensions pertinentes pour le clustering (Dim1, Dim2, Dim3)
data_3dim = data[["Dim1", "Dim2", "Dim3"]]

# from sklearn.neighbors import NearestNeighbors
# import matplotlib.pyplot as plt
# import numpy as np

# neigh = NearestNeighbors(n_neighbors=3)
# nbrs = neigh.fit(data_3dim)
# distances, indices = nbrs.kneighbors(data_3dim)
# distances = np.sort(distances, axis=0)
# distances = distances[:,1]
# plt.plot(distances)

rootClustersGM = tk.Tk()
rootClustersGM.title("Graphique t-SNE Cluster DBSCAN")
seaborn.set_style("white")

figClusterMG = plt.figure()
axClusterMG = figClusterMG.add_subplot(111, projection='3d')

colors = [
    'red', 'blue', 'green', 'orange', 'purple', 'cyan', 
    'magenta', 'yellow', 'brown', 'pink', 'lime', 'teal', 
    'navy', 'olive', 'maroon', 'gold', 'gray', 'black', 
    'violet', 'indigo'
]



from sklearn.mixture import GaussianMixture

# =============================================================================
# Variables pour GaussianMixture
# =============================================================================

nb_clusters = 4

n_components = nb_clusters  # Nombre de clusters (composants gaussiens)

# =============================================================================
# Changement de méthode de clustering : Gaussian Mixture Model
# =============================================================================

# Initialisation et application du modèle GMM
gmm = GaussianMixture(n_components=n_components, random_state=42)
data['Cluster'] = gmm.fit_predict(data_3dim)

# Ajouter les informations du cluster pour chaque point
cluster_data = []  # Réinitialiser pour GaussianMixture
for idx, row in data.iterrows():
    cluster_data.append([row['Cluster'], row['Dim1'], row['Dim2'], row['Dim3']])

# Créer un DataFrame avec les résultats
df_clusters = pandas.DataFrame(cluster_data, columns=["Cluster", "Dim1", "Dim2", "Dim3"])

# Remplir chaque liste de cluster avec les points associés
clusters = [[] for _ in range(n_components)]
for idx, row in df_clusters.iterrows():
    cluster_index = int(row['Cluster'])
    clusters[cluster_index].append([row['Dim1'], row['Dim2'], row['Dim3']])

# Calculer les centroids pour les clusters
centroids = []
for cluster in clusters:
    cluster = np.array(cluster)
    centroid = cluster.mean(axis=0)
    centroids.append(centroid)

# =============================================================================
# Visualisation (clusters, centres, forme)
# =============================================================================

for j, cluster in enumerate(clusters):
    cluster = np.array(cluster)
    axClusterMG.scatter(
        cluster[:, 0], cluster[:, 1], cluster[:, 2], 
        label=f'Cluster {j + 1}', color=colors[j], s=5)

if centroids:  # Si des centroids existent
    centroids = np.array(centroids)
    axClusterMG.scatter(
        centroids[:, 0], centroids[:, 1], centroids[:, 2], 
        label='Centre Cluster', color='black', s=100, marker='o')

# Ajouter des labels, un titre et une légende
axClusterMG.set_xlabel("Dimension 1")
axClusterMG.set_ylabel("Dimension 2")
axClusterMG.set_zlabel("Dimension 3")
axClusterMG.set_title("Plan factoriel")

# Ajouter une légende dynamique
axClusterMG.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Légende")

figClusterMG.savefig("/Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/resultats t-SNE/Plot-tSNE/Projection cluster k-means 3D.png", format='png', dpi=300)  # `dpi=300` pour une haute qualité

# Intégrer le graphique Matplotlib dans la fenêtre Tkinter
canvasCluster = FigureCanvasTkAgg(figClusterMG, master=rootClustersGM)
canvasCluster.draw()
canvasCluster.get_tk_widget().pack(expand=True, fill="both")

# Ajouter une barre d'outils de navigation (avec zoom et pan)
toolbarCluster = NavigationToolbar2Tk(canvasCluster, rootClustersGM)
toolbarCluster.update()
toolbarCluster.pack(side=tk.BOTTOM, fill=tk.X)

# Lancer l'application Tkinter
rootClustersGM.mainloop()

# =============================================================================
# Analyser le clustering - Barres empilées
# =============================================================================


# Grouper par 'Nom' et 'Cluster' et compter les points dans chaque groupe
cluster_counts = data.groupby(['Nom', 'Cluster']).size().reset_index(name='Point_Count')
pivot_table = cluster_counts.pivot(index='Nom', columns='Cluster', values='Point_Count').fillna(0)
pivot_table["total"] = pivot_table.sum(axis=1)
pivot_table.iloc[:, 0:9] = pivot_table.iloc[:, 0:9].div(pivot_table['total'], axis=0)

pivot_table.to_csv('/Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/pivot_table.csv', index=False)

# Supprimer la colonne "total" (non pertinente pour les barres empilées)
data_without_total = pivot_table.drop(columns=['total'])
# Ajouter une ligne de somme des colonnes
data_with_total = data_without_total.copy()
data_with_total.loc['Total'] = data_without_total.sum()

#noms_lignes = pivot_table.index.tolist()
# Tracer un graphique en barres empilées
plt.figure(figsize=(10, 6))
data_without_total.plot(kind='bar', stacked=True, figsize=(12, 8), cmap='tab20', width=0.8)

# Ajouter des titres et des labels
plt.title('Graphique en barres empilées des données', fontsize=16)
plt.xlabel('Noms', fontsize=12)
plt.ylabel('Valeurs', fontsize=12)
plt.xticks(rotation=45, fontsize=10)
plt.legend(title='Colonnes', bbox_to_anchor=(1.05, 1), loc='upper left')

# Afficher le graphique
plt.tight_layout()
plt.savefig("/Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/resultats t-SNE/Plot-tSNE/Analyser le clustering - Barres empilées.png", format='png', dpi=300)  # `dpi=300` pour une haute qualité
plt.show()


# =============================================================================
# Analyser le clustering - Diagrammes à Barres
# =============================================================================


totals = data_without_total.sum()
plt.figure(figsize=(10, 6))
plt.bar(totals.index, totals.values, color='skyblue', edgecolor='black')

# Ajouter des labels
plt.xlabel("Cluster")
plt.ylabel("Somme des valeurs")
plt.title("Diagramme à barres des sommes des colonnes")

# Annoter les barres avec les valeurs
for idx, value in enumerate(totals.values):
    plt.text(idx, value + 0.05, f"{value:.2f}", ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig("/Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/resultats t-SNE/Plot-tSNE/Analyser le clustering - Diagrammes à Barres.png", format='png', dpi=300)  # `dpi=300` pour une haute qualité
plt.show()


# =============================================================================
# Analyser le clustering - lignes connectées
# =============================================================================


plt.figure(figsize=(12, 6))

for idx, row in data_with_total.iterrows():
    plt.plot(data_without_total.columns, row.values, marker='o', label=idx)  

# Ajouter les labels et la légende
plt.xlabel("Colonnes")
plt.ylabel("Valeurs")
plt.title("Scatter plot avec lignes connectées")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Légende à droite
plt.tight_layout()
plt.savefig("/Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/resultats t-SNE/Plot-tSNE/Analyser le clustering - lignes connectées.png", format='png', dpi=300)  # `dpi=300` pour une haute qualité
plt.show()


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
# Sauvegarder les DataFrame dans un fichier CSV
# =============================================================================

output_path = '/Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/clusters_with_points_sklearn.csv'
df_clusters.to_csv(output_path, index=False)
