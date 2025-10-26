import os
import math
import pandas
import random
import seaborn
import numpy as np
import tkinter as tk
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import Birch
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from mpl_toolkits.mplot3d import Axes3D
from sklearn.mixture import GaussianMixture
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import AffinityPropagation
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# =============================================================================
# Variables
# =============================================================================

nb_clusters = 3
cluster_data = []  # DataFrame pour chaque point avec son cluster et ses dimensions
clusters = [[] for _ in range(nb_clusters)]  # Initialiser une liste vide pour contenir les clusters
data_cluster_tot = [[] for _ in range(4)]
data_new = []

Cluster1 = pandas.DataFrame()
Cluster1List = []
Cluster2 = pandas.DataFrame()
Cluster2List = []
Cluster3 = pandas.DataFrame()
Cluster3List = []
Cluster123 = pandas.DataFrame()
Cluster4 = pandas.DataFrame()
Cluster4List = []

# =============================================================================
# Intro de fenetre
# =============================================================================

data = pandas.read_csv('/Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/resultats t-SNE/Data_complet_ap_tSNE.csv')

noms_distincts = list(data["Nom"].unique())
N = len(noms_distincts)

# Sélectionner les dimensions pertinentes pour le clustering (Dim1, Dim2, Dim3)
data_3dim = data[["Dim1", "Dim2", "Dim3"]].values

rootClusterSimpleLinkage = tk.Tk()
rootClusterSimpleLinkage.title("Graphique t-SNE Cluster Simple Linkage")
seaborn.set_style("white")

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
df_clusters = pandas.DataFrame(cluster_data, columns=["Cluster", "Dim1", "Dim2", "Dim3", "Nom"])

# Remplir chaque liste de cluster avec les points associés
for idx, row in df_clusters.iterrows():
    cluster_index = int(row['Cluster']) - 1  # Indice du cluster (1 à nb_clusters)
    clusters[cluster_index].append([row['Dim1'], row['Dim2'], row['Dim3'], row["Nom"]])  # Ajouter les coordonnées du point

for idx, row in df_clusters.iterrows():
    new_row = {'Dim1': row['Dim1'], 'Dim2': row['Dim2'], 'Dim3': row['Dim3'], 'Nom': row["Nom"]}
    new_df = pandas.DataFrame([new_row])
    if row['Cluster']==2:
        Cluster4 = pandas.concat([Cluster4, new_df], ignore_index=True)
        Cluster4List = Cluster4.to_numpy()
    else:
        Cluster123 = pandas.concat([Cluster123, new_df], ignore_index=True)

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
    new_df = pandas.DataFrame([new_row])
    if row['Cluster123']==0:
        Cluster1 = pandas.concat([Cluster1, new_df], ignore_index=True)
        Cluster1List = Cluster1.to_numpy()
    elif row['Cluster123']==1:
        Cluster2 = pandas.concat([Cluster2, new_df], ignore_index=True)
        Cluster2List = Cluster2.to_numpy()
    elif row['Cluster123']==2:
        Cluster3 = pandas.concat([Cluster3, new_df], ignore_index=True)
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
Data_new = pandas.DataFrame(data_new, columns=['Dim1', 'Dim2', 'Dim3', 'Cluster', 'Nom'])

    
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

# #noms_lignes = pivot_table.index.tolist()
# # Tracer un graphique en barres empilées
# plt.figure(figsize=(10, 6))
# data_without_total.plot(kind='bar', stacked=True, figsize=(12, 8), cmap='tab20', width=0.8)

# # Ajouter des titres et des labels
# plt.title('Graphique en barres empilées des données', fontsize=16)
# plt.xlabel('Noms', fontsize=12)
# plt.ylabel('Valeurs', fontsize=12)
# plt.xticks(rotation=45, fontsize=10)
# plt.legend(title='Colonnes', bbox_to_anchor=(1.05, 1), loc='upper left')

# # Afficher le graphique
# plt.tight_layout()
# plt.savefig("/Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/resultats t-SNE/Plot-tSNE/Analyser le clustering - Barres empilées.png", format='png', dpi=300)  # `dpi=300` pour une haute qualité
# plt.show()


# =============================================================================
# Analyser le clustering - Diagrammes à Barres
# =============================================================================


# totals = data_without_total.sum()
# plt.figure(figsize=(10, 6))
# plt.bar(totals.index, totals.values, color='skyblue', edgecolor='black')

# # Ajouter des labels
# plt.xlabel("Cluster")
# plt.ylabel("Somme des valeurs")
# plt.title("Diagramme à barres des sommes des colonnes")

# # Annoter les barres avec les valeurs
# for idx, value in enumerate(totals.values):
#     plt.text(idx, value + 0.05, f"{value:.2f}", ha='center', va='bottom', fontsize=9)

# plt.tight_layout()
# plt.savefig("/Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/resultats t-SNE/Plot-tSNE/Analyser le clustering - Diagrammes à Barres.png", format='png', dpi=300)  # `dpi=300` pour une haute qualité
# plt.show()


# =============================================================================
# Analyser le clustering - lignes connectées
# =============================================================================


# plt.figure(figsize=(12, 6))

# for idx, row in data_with_total.iterrows():
#     plt.plot(data_without_total.columns, row.values, marker='o', label=idx)  

# # Ajouter les labels et la légende
# plt.xlabel("Colonnes")
# plt.ylabel("Valeurs")
# plt.title("Scatter plot avec lignes connectées")
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Légende à droite
# plt.tight_layout()
# plt.savefig("/Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/resultats t-SNE/Plot-tSNE/Analyser le clustering - lignes connectées.png", format='png', dpi=300)  # `dpi=300` pour une haute qualité
# plt.show()


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
distance_df = pandas.DataFrame(
    distances,
    columns=[f'Person_{i}' for i in range(len(coordinates))],
    index=[f'Person_{i}' for i in range(len(coordinates))]
)

# Appliquer l'algorithme K-means
n_clusters2 = 4  # Nombre de clusters souhaités
kmeans = KMeans(n_clusters=n_clusters2, random_state=42)
kmeans.fit(coordinates)

data_without_total.to_csv('/Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/data_without_total1.csv', index=False)


# Ajouter les labels des clusters au DataFrame
data_without_total['Cluster'] = kmeans.labels_


root = tk.Tk()
root.title("Graphique")
seaborn.set_style("white")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Visualisation des clusters (si possible, avec 2 dimensions principales)
scatter = ax.scatter(data_without_total.iloc[0:N, 0], data_without_total.iloc[0:N, 1], data_without_total.iloc[0:N, 2], c=kmeans.labels_, cmap='viridis')
# Ajouter les étiquettes des axes
ax.set_title('K-means Clustering (3D)')
ax.set_xlabel('Component 1')
ax.set_ylabel('Component 2')
ax.set_zlabel('Component 3')

# Ajouter une barre de couleurs
fig.colorbar(scatter, ax=ax, label='Cluster')


# Ajouter les noms des personnes à côté des points
for i, name in enumerate(data_without_total.index.to_list()):
    ax.text(coordinates[i, 0], coordinates[i, 1], coordinates[i, 2], name, fontsize=8)


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
# =============================================================================
# Sauvegarder les DataFrame dans un fichier CSV
# =============================================================================

#output_path = '/Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/clusters_with_points_sklearn.csv'
#df_clusters.to_csv(output_path, index=False)
