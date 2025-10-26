# FRENCH VERSION BELOW
# Runner-s-Performance

Analysis and Prediction of Runners’ Performances

## 🏃 Project Overview

**RunAI** is a project focused on the **analysis and prediction of athletic performance**, specifically for **running and trail running**.  
Its goal is to provide an intelligent platform that enables runners to **track their progress**, **analyze their performances**, and **predict their future results** using **machine learning algorithms**.

### Main Features

- **Runner Clustering** based on performance profiles  
- **Prediction of future speeds and records** using time series models (ARIMA/SARIMAX)  
- **Personalized race recommendations** adapted to each runner’s level  
- **Interactive graphical interface** for exploring data and results  

---

## 📂 Repository Structure

| File                                                                        | Description                                                        |
| --------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| `Total-PI-Final.py`                                                         | **Main script**: complete project pipeline                         |
| `InterfaceFinal.py`                                                         | Tkinter GUI for interacting with results                           |
| `decompresserGPX.py`                                                        | Decompression and reading of `.gpx` or `.fit.gz` files             |
| `TopographieCourse.py`                                                      | Analysis of race elevation profiles                                |
| `Evolution-Pace-Distance-1Graph.py`                                         | Visualization of pace evolution over distance                      |
| `Evolution-Pace-Distance-NGraph.py`                                         | Multi-runner pace evolution analysis                               |
| `Evolution_Pace_MultiFig.py`                                                | Comparison of multiple runners’ progressions                       |
| `Algo-Clustering.py`                                                        | Central script combining different clustering algorithms           |
| `Algo-k-means-tSNE.py`, `Algo-DBSCAN-tSNE.py`, `Algo-Gaussian-tSNE.py`, etc. | Clustering scripts combining various algorithms with t-SNE         |
| `Algo-k-means-ACP.py`                                                       | Clustering after dimensionality reduction with PCA                 |
| `README.md`                                                                 | This file 😉                                                       |

---

## 🔬 Methodology

### 1. Data Collection and Preparation

Data is sourced from **Strava** in `.csv` or `.gpx` formats.  
It includes distance, pace, average speed, time, elevation gain, and more.  
All files are cleaned, standardized, and normalized to ensure consistent analysis.

### 2. Clustering and Classification

Runners are grouped based on their performance profiles using:

- **K-Means**
- **DBSCAN**
- **Gaussian Mixture Model**
- **Agglomerative Clustering**

Dimensionality reduction is applied via **PCA** or **t-SNE** for intuitive 2D/3D visualization.

### 3. Performance Prediction

Using **ARIMA/SARIMAX** and **polynomial regression** models, the system predicts:

- The evolution of average speed over time  
- Future performance records on 5K, 10K, half-marathon, and marathon distances  

### 4. Race Recommendation

Based on each runner’s cluster and performance level, the system suggests races tailored to their profile.

### 5. Graphical Interface

A user-friendly interface allows for the exploration of:

- Runner clusters and projections  
- Predicted performance evolution  
- Temporal and topographic visualizations  

---

## 📊 Technologies Used

| Domain                   | Tools / Libraries             |
| ------------------------- | ----------------------------- |
| **Main language**         | Python                        |
| **Data analysis**         | Pandas, NumPy                 |
| **Visualization**         | Matplotlib, Seaborn           |
| **Machine Learning**      | Scikit-Learn                  |
| **Time Series**           | Statsmodels (ARIMA, SARIMAX)  |
| **Graphical Interface**   | Tkinter                       |
| **GPS File Handling**     | GPXPy, gzip                   |

---

## ✅ Key Results

- **Meaningful segmentation** of runners into performance-based clusters  
- **Clear visualizations** through dimensionality reduction (t-SNE, PCA)  
- **Accurate performance predictions** over time  
- **Functional GUI** for exploration and personalized race recommendations  

---

## ⚠️ Challenges

- Heterogeneous data formats (`.fit`, `.gpx`, missing values) across users  
- Limited dataset size, affecting model robustness  
- Lack of geolocated data for nearby race recommendations  

---

## 🚀 Future Work

- Full integration of `.fit` and `.fit.gz` file formats  
- Extension to **trail running** and other sports  
- Improved predictive performance with **deep learning (LSTM)** models  
- Addition of a **personalized training tracking module**  

---

## 👤 Author

**Camille Auvity**  
Data Science Student – Mines Saint-Étienne x EM Lyon  
📧 (mailto:caauvity@orange.fr)

------------------------------------------------------
# FRENCH VERSION
# Runner-s-Performance

Analyse et Prédiction des Performances des Coureurs

## Description du projet

**RunAI** est un projet d’analyse et de prédiction de performances sportives, axé sur la **course à pied et le trail**.
L’objectif est de fournir une plateforme intelligente permettant aux coureurs de **suivre leurs progrès**, **analyser leurs performances**, et **prédire leurs résultats futurs** grâce à des **algorithmes de machine learning**.

Le projet combine plusieurs volets :

* **Clustering des coureurs** selon leur profil de performance.
* **Prédiction des vitesses et records futurs** à l’aide de modèles de séries temporelles (ARIMA/SARIMAX).
* **Recommandation de courses adaptées** au niveau de chaque coureur.
* **Interface graphique interactive** facilitant l’exploration des résultats.

## 📂 Structure du dépôt

| Fichier                                                                      | Description                                                        |
| ---------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| `Total-PI-Final.py`                                                          | **Fichier principal** : pipeline complet du projet                 |
| `InterfaceFinal.py`                                                          | Interface graphique Tkinter pour interagir avec les résultats      |
| `decompresserGPX.py`                                                         | Décompression et lecture des fichiers `.gpx` ou `.fit.gz`          |
| `TopographieCourse.py`                                                       | Analyse du profil altimétrique des courses                         |
| `Evolution-Pace-Distance-1Graph.py`                                          | Visualisation de l’évolution de l’allure selon la distance         |
| `Evolution-Pace-Distance-NGraph.py`                                          | Analyse multi-coureurs de l’évolution du pace                      |
| `Evolution_Pace_MultiFig.py`                                                 | Comparaison des progressions sur plusieurs figures                 |
| `Algo-Clustering.py`                                                         | Script central regroupant les différents algorithmes de clustering |
| `Algo-k-means-tSNE.py`, `Algo-DBSCAN-tSNE.py`, `Algo-Gaussian-tSNE.py`, etc. | Scripts dédiés à chaque méthode de clustering couplée à t-SNE      |
| `Algo-k-means-ACP.py`                                                        | Clustering après réduction de dimension par ACP                    |
| `README.md`                                                                  | Ce fichier 😉                                                      |

##  Méthodologie

### 1 Collecte et préparation des données

Les données proviennent de **Strava** sous forme de fichiers `.csv` ou `.gpx`.
Elles contiennent : distance, allure, vitesse moyenne, temps, D+, etc.
Les fichiers sont nettoyés, homogénéisés et normalisés pour permettre les analyses.

### 2 Clustering et classification

Les coureurs sont regroupés selon leurs profils via différentes méthodes :

* **K-Means**
* **DBSCAN**
* **Gaussian Mixture Model**
* **Agglomerative Clustering**

La réduction de dimension est effectuée avec **ACP** ou **t-SNE**, facilitant la visualisation 2D/3D.

### 3 Prédiction des performances

Les modèles **ARIMA/SARIMAX** et **régressions polynomiales** sont utilisés pour prédire :

* L’évolution de la vitesse moyenne dans le temps
* Les records futurs sur 5 km, 10 km, semi-marathon, marathon

### 4 Recommandation de courses

À partir du profil de chaque coureur (cluster et niveau), le système suggère une course adaptée à son niveau de performance.

### 5 Interface graphique

Une interface simple permet d’explorer :

* Les clusters et projections des coureurs
* Les prédictions de performances futures
* Les visualisations temporelles et topographiques

---

## 📊 Technologies utilisées

| Domaine                  | Outils / Bibliothèques       |
| ------------------------ | ---------------------------- |
| **Langage principal**    | Python                       |
| **Analyse de données**   | Pandas, NumPy                |
| **Visualisation**        | Matplotlib, Seaborn          |
| **Machine Learning**     | Scikit-Learn                 |
| **Séries temporelles**   | Statsmodels (ARIMA, SARIMAX) |
| **Interface graphique**  | Tkinter                      |
| **Gestion fichiers GPS** | GPXPy, gzip                  |

## Résultats clés

* Segmentation des coureurs en **niveaux cohérents** selon leurs performances.
* **Visualisations claires** grâce à la réduction de dimension (t-SNE).
* **Prédictions réalistes** de l’évolution des performances dans le temps.
* **Interface fonctionnelle** pour explorer les résultats et recommandations.

## Difficultés rencontrées

* Données hétérogènes selon les utilisateurs (formats `.fit`, `.gpx`, valeurs manquantes).
* Volume de données limité réduisant la robustesse des modèles.
* Manque de données géolocalisées pour la recommandation de courses proches.

## Perspectives d’évolution

* Intégration complète des fichiers `.fit` et `.fit.gz`.
* Extension au **trail** et à d’autres disciplines sportives.
* Amélioration des modèles prédictifs via des approches deep learning (LSTM).
* Ajout d’un module de **suivi d’entraînement personnalisé**.

## 👤 Auteur

**Camille Auvity**
Étudiant en Data Science – Mines Saint-Étienne x EM Lyon
📧 (mailto:caauvity@orange.fr)
