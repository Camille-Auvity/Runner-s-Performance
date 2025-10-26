# Runner-s-Performance
## English Version below

Analyse et Prédiction des Performances des Coureurs

## Description du projet

**RunAI** est un projet d’analyse et de prédiction de performances sportives, axé sur la **course à pied et le trail**.
L’objectif est de fournir une plateforme intelligente permettant aux coureurs de **suivre leurs progrès**, **analyser leurs performances**, et **prédire leurs résultats futurs** grâce à des **algorithmes de machine learning**.

Le projet combine plusieurs volets :

* **Clustering des coureurs** selon leur profil de performance.
* **Prédiction des vitesses et records futurs** à l’aide de modèles de séries temporelles (ARIMA/SARIMAX).
* **Recommandation de courses adaptées** au niveau de chaque coureur.
* **Interface graphique interactive** facilitant l’exploration des résultats.

## 🚀 Lancer le projet

### 🧩 Prérequis

Avant de lancer le projet, assure-toi d’avoir installé les dépendances suivantes :

```bash
pip install numpy pandas matplotlib seaborn scikit-learn statsmodels tkinter
```

---

### ▶️ Exécution

Le fichier principal à exécuter est :

```bash
python Total-PI-Final.py
```

Ce script orchestre l’ensemble du pipeline :

* Chargement et nettoyage des données Strava
* Application des algorithmes de clustering (K-Means, DBSCAN, Gaussian Mixture, etc.)
* Visualisation 2D/3D via t-SNE ou ACP
* Prédiction des performances à partir des historiques
* Lancement de l’interface utilisateur (Tkinter)

---

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

---

## 🧠 Méthodologie

### 1️⃣ Collecte et préparation des données

Les données proviennent de **Strava** sous forme de fichiers `.csv` ou `.gpx`.
Elles contiennent : distance, allure, vitesse moyenne, temps, D+, etc.
Les fichiers sont nettoyés, homogénéisés et normalisés pour permettre les analyses.

### 2️⃣ Clustering et classification

Les coureurs sont regroupés selon leurs profils via différentes méthodes :

* **K-Means**
* **DBSCAN**
* **Gaussian Mixture Model**
* **Agglomerative Clustering**

La réduction de dimension est effectuée avec **ACP** ou **t-SNE**, facilitant la visualisation 2D/3D.

### 3️⃣ Prédiction des performances

Les modèles **ARIMA/SARIMAX** et **régressions polynomiales** sont utilisés pour prédire :

* L’évolution de la vitesse moyenne dans le temps
* Les records futurs sur 5 km, 10 km, semi-marathon, marathon

### 4️⃣ Recommandation de courses

À partir du profil de chaque coureur (cluster et niveau), le système suggère une course adaptée à son niveau de performance.

### 5️⃣ Interface graphique

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

---

## 💡 Résultats clés

* Segmentation des coureurs en **niveaux cohérents** selon leurs performances.
* **Visualisations claires** grâce à la réduction de dimension (t-SNE).
* **Prédictions réalistes** de l’évolution des performances dans le temps.
* **Interface fonctionnelle** pour explorer les résultats et recommandations.

---

## ⚙️ Difficultés rencontrées

* Données hétérogènes selon les utilisateurs (formats `.fit`, `.gpx`, valeurs manquantes).
* Volume de données limité réduisant la robustesse des modèles.
* Manque de données géolocalisées pour la recommandation de courses proches.

---

## 🔮 Perspectives d’évolution

* Intégration complète des fichiers `.fit` et `.fit.gz`.
* Extension au **trail** et à d’autres disciplines sportives.
* Amélioration des modèles prédictifs via des approches deep learning (LSTM).
* Ajout d’un module de **suivi d’entraînement personnalisé**.

---

## 👤 Auteur

**Camille Auvity**
Étudiant en Data Science – Mines Saint-Étienne x EM Lyon
📧 [[camille.auvity@example.com](mailto:camille.auvity@example.com)] 
