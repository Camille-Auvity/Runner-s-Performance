#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 11:14:23 2025

@author: camilleauvity
"""

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import folium
import gpxpy
import webbrowser

# Chargement des données
tableau_proximite = pd.read_csv("/Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/Evo-Pace-Selon-Distance/tableau_proximite.csv")
data_without_total = pd.read_csv('/Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/data_without_total1.csv')
data_cleaned_complet = pd.read_csv('/Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/data_cleaned_complet.csv')

with open("/Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/noms_lignes.txt", "r") as f:
    noms_lignes = [line.strip() for line in f.readlines()]

data_without_total.index = noms_lignes

data = pd.read_csv('/Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/resultats t-SNE/Data_complet_ap_tSNE.csv')
noms_distincts = list(data["Nom"].unique())
N = len(noms_distincts)

# Chemins des images associés aux prénoms
images_utilisateurs = {
    "Camille": "/Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/Evolution Pace/Evolution_Pace_Camille.png",
    "Maxime": "/Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/Evolution Pace/Evolution_Pace_Maxime.png",
    "Oscar": "/Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/Evolution Pace/Evolution_Pace_Oscar.png",
    "Charline": "/Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/Evolution Pace/Evolution_Pace_Charline.png",
}

# Variable globale
prenom_utilisateur = None

def afficher_interface_principale():
    global prenom_utilisateur
    global data_cleaned_complet

    root = tk.Tk()
    root.title("Application d'Analyse")
    root.geometry("1600x900")
    root.configure(bg="#f4f4f4")

    style = ttk.Style()
    style.configure("TButton", font=("Arial", 12), padding=10)

    from tkinter import messagebox

    def demander_record():
        data = pd.read_csv('/Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/resultats t-SNE/Data_complet_ap_tSNE.csv')
        data_without_total = pd.read_csv('/Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/data_without_total1.csv')
    
        with open("/Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/noms_lignes.txt", "r") as f:
            noms_lignes = [line.strip() for line in f.readlines()]
    
        data_without_total.index = noms_lignes
        
        noms_distincts = list(data["Nom"].unique())
        N = len(noms_distincts)
    
        coordinates = data_without_total.iloc[0:N, :].to_numpy()
        distances = squareform(pdist(coordinates, metric='euclidean'))
        kmeans = KMeans(n_clusters=4, random_state=42)
        kmeans.fit(coordinates)
        data_without_total['Cluster'] = kmeans.labels_ 
        
        def valider_record():
            global noms_distincts
            global kmeans
            record_choisi = combo_record.get()
            if record_choisi:
                # Trouver la position de l'utilisateur dans noms_distincts
                if prenom_utilisateur in noms_distincts:
                    index_personne = noms_distincts.index(prenom_utilisateur)
                    #niveau_personne = kmeans.labels_[index_personne]  # Récupérer le niveau à partir des clusters
                    niveau_personne = data_without_total.loc[prenom_utilisateur, "Cluster"]
                    # Messages en fonction du niveau
                    messages_niveaux = {
                        0: f"Vous êtes débutant. Ce choix est parfait pour débuter votre progression. {combo_record.get()}",
                        1: "Vous êtes intermédiaire. C'est un excellent objectif pour vous dépasser.",
                        2: "Vous êtes avancé. Vous êtes prêt pour relever ce défi ambitieux.",
                        3: "Vous êtes expert. Cet objectif est parfait pour perfectionner vos performances."
                    }
    
                    message = messages_niveaux.get(niveau_personne, "Votre niveau n'a pas été identifié correctement.")
                    #tk.messagebox.showinfo("Choix validé", f"Vous souhaitez améliorer votre record sur : {record_choisi}\n{message}")
                    
                    
                    if record_choisi == "5 km" and niveau_personne == 0:
                        tk.messagebox.showinfo(
                            "Programme Running Débutant - 5 km",
                            """
                            5 km : Objectif - Courir 5 km en 6 semaines
                            Fréquence : 3 fois par semaine
                            Focus : Endurance, régularité, et rythme modéré.
                    
                            Semaine 1 :
                              Jour 1 : 1 min de course / 1 min de marche, répéter 10 fois
                              Jour 2 : 20 min de marche rapide
                              Jour 3 : 2 min de course / 1 min de marche, répéter 7 fois
                    
                            Semaine 2 :
                              Jour 1 : 3 min de course / 1 min de marche, répéter 6 fois
                              Jour 2 : 20 min de marche rapide
                              Jour 3 : 4 min de course / 1 min de marche, répéter 5 fois
                    
                            Semaine 3 :
                              Jour 1 : 5 min de course / 1 min de marche, répéter 4 fois
                              Jour 2 : 20 min de marche rapide ou 15 min de course lente
                              Jour 3 : 7 min de course / 1 min de marche, répéter 3 fois
                    
                            Semaine 4 :
                              Jour 1 : 10 min de course, 1 min de marche, répéter 2 fois
                              Jour 2 : 20 min de marche rapide ou 15 min de course lente
                              Jour 3 : 12 min de course, 1 min de marche, répéter 2 fois
                    
                            Semaine 5 :
                              Jour 1 : 15 min de course, 1 min de marche, puis 5 min de course
                              Jour 2 : 20 min de course lente
                              Jour 3 : 20 min de course, 1 min de marche, puis 5 min de course
                    
                            Semaine 6 :
                              Jour 1 : 20 min de course
                              Jour 2 : 25 min de course lente
                              Jour 3 : Testez-vous : 5 km à votre rythme. 🎉
                            """
                        )

                    if record_choisi == "5 km" and niveau_personne == 1:
                        tk.messagebox.showinfo(
                            "Programme Running Intermédiaire - 5 km",
                            """
                            5 km : Objectif - Améliorer son temps en 6 semaines
                            Fréquence : 3 à 4 fois par semaine
                            Focus : Vitesse, régularité, et fractionné.
                    
                            Semaine 1-2 :
                              - Une sortie de 3 à 4 km en course lente
                              - Une séance fractionnée : 1 min rapide / 1 min lent (répéter 10 fois)
                              - Une sortie longue : 5 à 6 km à rythme modéré
                              - Une séance de récupération : 20 min de course lente ou marche rapide
                    
                            Semaine 3-4 :
                              - Une sortie modérée : 4 à 5 km à rythme constant
                              - Une séance fractionnée : 2 min rapide / 1 min lent (répéter 8 fois)
                              - Une sortie longue : 6 à 7 km à rythme modéré
                              - Une séance de récupération : 20-25 min en course lente
                    
                            Semaine 5-6 :
                              - Une sortie modérée : 5 km à rythme constant
                              - Une séance fractionnée : 3 min rapide / 1 min lent (répéter 6 fois)
                              - Une sortie longue : 7 à 8 km à rythme modéré
                              - Testez-vous : Courrez un 5 km à votre meilleur rythme. 🎉
                            """
                        )

                    if record_choisi == "5 km" and niveau_personne == 2:
                        tk.messagebox.showinfo(
                            "Programme Running Avancé - 5 km",
                            """
                            5 km : Objectif - Maximiser la performance en 6 semaines
                            Fréquence : 4 à 5 fois par semaine
                            Focus : Vitesse, fractionné, et intensité.
                    
                            Semaine 1-2 :
                              - Une sortie rapide : 5 km à 80-85% de votre effort max
                              - Une séance fractionnée : 400 m rapide / 200 m récupération (répéter 8 fois)
                              - Une sortie longue : 6-7 km à rythme modéré
                              - Une séance de récupération : 30 min en course lente
                    
                            Semaine 3-4 :
                              - Une sortie tempo : 6 km à 85-90% de votre effort max
                              - Une séance fractionnée : 800 m rapide / 400 m récupération (répéter 6 fois)
                              - Une sortie longue : 7-8 km à rythme modéré
                              - Une séance de récupération : 35 min en course lente
                    
                            Semaine 5-6 :
                              - Une sortie rapide : 5 km à 90% de votre effort max
                              - Une séance fractionnée : 1 000 m rapide / 500 m récupération (répéter 5 fois)
                              - Une sortie longue : 8-9 km à rythme modéré
                              - Testez-vous : Courez un 5 km à votre meilleur rythme. 🎉
                            """
                        )

                    if record_choisi == "5 km" and niveau_personne == 3:
                        tk.messagebox.showinfo(
                            "Programme Running Expert - 5 km",
                            """
                            5 km : Objectif - Battre un record personnel en 6 semaines
                            Fréquence : 5 à 6 fois par semaine
                            Focus : Vitesse maximale, gestion des allures, et optimisation.
                    
                            Semaine 1-2 :
                              - Séance 1 : 6x1 000 m à 90-95% VMA, récupération 90 sec
                              - Séance 2 : 8 km à allure seuil (80-85% VMA)
                              - Séance 3 : 12 km en endurance fondamentale
                              - Séance 4 : 10x400 m à 100% VMA, récupération 200 m lent
                              - Séance 5 : 30 min en course lente ou repos actif
                    
                            Semaine 3-4 :
                              - Séance 1 : 4x1 500 m à allure 10 km, récupération 2 min
                              - Séance 2 : 10 km à allure seuil + 4 km en endurance
                              - Séance 3 : 14 km en endurance fondamentale
                              - Séance 4 : 12x300 m à 100-110% VMA, récupération 100 m
                              - Séance 5 : 40 min de course lente
                    
                            Semaine 5-6 :
                              - Séance 1 : 8x800 m à 95% VMA, récupération 2 min
                              - Séance 2 : 12 km à allure seuil
                              - Séance 3 : 16 km en endurance fondamentale
                              - Séance 4 : 6x1 000 m à 90% VMA
                              - Testez-vous : Courez un 5 km à votre meilleur rythme. 🎉
                            """
                        )

                    elif record_choisi == "10 km" and niveau_personne == 0:
                        tk.messagebox.showinfo(
                            "Programme Running Débutant - 10 km",
                            """
                            10 km : Objectif - Courir 10 km en 10 semaines
                            Fréquence : 3 à 4 fois par semaine
                            Focus : Endurance, régularité, et augmentation progressive de la distance.
                    
                            Semaine 1-2 :
                              3 sorties par semaine
                              Exemple : 2 km de course lente + 1 min de marche (répéter 2 fois)
                              Une sortie supplémentaire en marche rapide (20-30 min)
                    
                            Semaine 3-4 :
                              3 sorties par semaine
                              Exemple : 3 km de course lente + 2 min de marche (répéter 2 fois)
                              Une sortie longue : 5 km en course lente
                    
                            Semaine 5-6 :
                              3-4 sorties par semaine
                              Exemple : Alternez entre 4 km à rythme modéré et 1 min de marche
                              Une sortie longue : 7 km en course lente
                    
                            Semaine 7-8 :
                              4 sorties par semaine
                              Exemple : 5 km de course lente + 2 min de marche
                              Une sortie longue : 8-9 km à rythme modéré
                    
                            Semaine 9-10 :
                              4 sorties par semaine
                              Exemple : 6 km à rythme modéré
                              Une sortie longue : 10 km en course lente ou à votre rythme. 🎉
                            """
                        )

                    if record_choisi == "10 km" and niveau_personne == 1:
                        tk.messagebox.showinfo(
                            "Programme Running Intermédiaire - 10 km",
                            """
                            10 km : Objectif - Courir plus vite ou plus confortablement en 8 semaines
                            Fréquence : 3 à 4 fois par semaine
                            Focus : Vitesse, régularité, et sorties longues.
                    
                            Semaine 1-2 :
                              - Une sortie modérée : 5 à 6 km
                              - Une séance fractionnée : 1 min rapide / 1 min lent (répéter 10 fois)
                              - Une sortie longue : 7 à 8 km en course lente
                              - Une séance de récupération : 25 min de course lente
                    
                            Semaine 3-4 :
                              - Une sortie modérée : 6 à 7 km
                              - Une séance fractionnée : 2 min rapide / 1 min lent (répéter 8 fois)
                              - Une sortie longue : 8 à 10 km à rythme modéré
                              - Une séance de récupération : 30 min en course lente
                    
                            Semaine 5-6 :
                              - Une sortie modérée : 7 à 8 km
                              - Une séance fractionnée : 3 min rapide / 1 min lent (répéter 6 fois)
                              - Une sortie longue : 10 à 12 km à rythme modéré
                              - Une séance de récupération : 30-35 min en course lente
                    
                            Semaine 7-8 :
                              - Une sortie modérée : 8 à 9 km
                              - Une sortie longue : 10 à 12 km à rythme modéré
                              - Testez-vous : Courez un 10 km à votre meilleur rythme. 🎉
                            """
                        )

                    if record_choisi == "10 km" and niveau_personne == 2:
                        tk.messagebox.showinfo(
                            "Programme Running Avancé - 10 km",
                            """
                            10 km : Objectif - Atteindre une performance optimale en 8 semaines
                            Fréquence : 5 fois par semaine
                            Focus : Fractionné, tempo, et endurance.
                    
                            Semaine 1-2 :
                              - Une sortie tempo : 6-7 km à 85% de votre effort max
                              - Une séance fractionnée : 400 m rapide / 200 m récupération (répéter 10 fois)
                              - Une sortie longue : 8-10 km à rythme modéré
                              - Une séance de récupération : 30 min en course lente
                    
                            Semaine 3-4 :
                              - Une sortie tempo : 8 km à 85-90% de votre effort max
                              - Une séance fractionnée : 800 m rapide / 400 m récupération (répéter 8 fois)
                              - Une sortie longue : 10-12 km à rythme modéré
                              - Une séance de récupération : 35 min en course lente
                    
                            Semaine 5-6 :
                              - Une sortie tempo : 9 km à 90% de votre effort max
                              - Une séance fractionnée : 1 000 m rapide / 500 m récupération (répéter 6 fois)
                              - Une sortie longue : 12-14 km à rythme modéré
                              - Une séance de récupération : 40 min en course lente
                    
                            Semaine 7-8 :
                              - Une sortie tempo : 10 km à 90% de votre effort max
                              - Une sortie longue : 14 km à rythme modéré
                              - Testez-vous : Courrez un 10 km à votre meilleur rythme. 🎉
                            """
                        )

                    if record_choisi == "10 km" and niveau_personne == 3:
                        tk.messagebox.showinfo(
                            "Programme Running Expert - 10 km",
                            """
                            10 km : Objectif - Battre un record personnel en 8 semaines
                            Fréquence : 6 fois par semaine
                            Focus : Vitesse, seuil, et endurance.
                    
                            Semaine 1-2 :
                              - Séance 1 : 5x2 000 m à allure 10 km, récupération 2 min
                              - Séance 2 : 12 km à allure seuil (80-85% VMA)
                              - Séance 3 : 16 km en endurance fondamentale
                              - Séance 4 : 10x400 m à 100% VMA, récupération 200 m
                              - Séance 5 : 45 min en course lente ou repos actif
                              - Séance 6 : 18 km en endurance lente
                    
                            Semaine 3-6 :
                              - Séance 1 : 6x1 500 m à 90-95% VMA, récupération 2 min
                              - Séance 2 : 14 km à allure seuil + 4 km à allure modérée
                              - Séance 3 : 20 km en endurance fondamentale
                              - Séance 4 : 12x300 m à 110% VMA, récupération 100 m
                              - Séance 5 : 60 min en course lente
                              - Séance 6 : 20-22 km en endurance lente
                    
                            Semaine 7-8 :
                              - Réduction progressive pour récupération.
                              - Testez-vous : Courrez un 10 km à votre meilleur rythme. 🎉
                            """
                        )

                    elif record_choisi == "21 km (semi-marathon)" and niveau_personne == 0:
                        tk.messagebox.showinfo(
                            "Programme Running Débutant - 21 km",
                            """
                            Semi-marathon : Objectif - Courir 21 km en 12 semaines
                            Fréquence : 3 à 5 fois par semaine
                            Focus : Endurance, gestion de l'énergie, et sorties longues.
                    
                            Semaine 1-4 :
                              3 à 4 sorties par semaine
                              - Exemple : 5 à 8 km de course lente
                              - Une sortie longue : 10 à 12 km en course lente
                              - Travaillez la régularité.
                    
                            Semaine 5-8 :
                              4 sorties par semaine
                              - Une sortie modérée : 8 à 10 km
                              - Une sortie longue : 14 à 16 km à rythme lent
                              - Séance fractionnée : 2 min rapide / 1 min lent (répéter 5 fois)
                    
                            Semaine 9-10 :
                              4 à 5 sorties par semaine
                              - Une sortie longue : 18 à 19 km en course lente
                              - Une sortie modérée : 10 à 12 km
                              - Séance de vitesse : 4 min rapide / 2 min lent (répéter 4 fois)
                    
                            Semaine 11-12 :
                              3 à 4 sorties par semaine (réduction progressive)
                              - Une sortie modérée : 8 à 10 km
                              - Une sortie longue : 15 km en course lente (dernière semaine)
                              - Jour J : Semi-marathon (21 km). 🎉
                            """
                        )

                    if record_choisi == "21 km (semi-marathon)" and niveau_personne == 1:
                        tk.messagebox.showinfo(
                            "Programme Running Intermédiaire - 21 km",
                            """
                            Semi-marathon : Objectif - Améliorer votre temps en 10 semaines
                            Fréquence : 4 à 5 fois par semaine
                            Focus : Endurance, vitesse, et gestion de l'effort.
                    
                            Semaine 1-4 :
                              - Une sortie modérée : 8 à 10 km
                              - Une séance fractionnée : 2 min rapide / 1 min lent (répéter 8 fois)
                              - Une sortie longue : 12 à 15 km en course lente
                              - Une séance de récupération : 30 min en course lente
                    
                            Semaine 5-8 :
                              - Une sortie modérée : 10 à 12 km
                              - Une séance fractionnée : 3 min rapide / 2 min lent (répéter 6 fois)
                              - Une sortie longue : 15 à 18 km à rythme modéré
                              - Une séance de récupération : 35-40 min en course lente
                    
                            Semaine 9-10 :
                              - Une sortie modérée : 12 à 15 km
                              - Une sortie longue : 18 à 21 km en course lente
                              - Testez-vous : Courez un semi-marathon (21 km) à votre meilleur rythme. 🎉
                            """
                        )

                    if record_choisi == "21 km (semi-marathon)" and niveau_personne == 2:
                        tk.messagebox.showinfo(
                            "Programme Running Avancé - 21 km",
                            """
                            Semi-marathon : Objectif - Réaliser un record personnel en 10 semaines
                            Fréquence : 5 à 6 fois par semaine
                            Focus : Endurance, vitesse, et gestion de l'effort.
                    
                            Semaine 1-4 :
                              - Une sortie tempo : 10-12 km à 85% de votre effort max
                              - Une séance fractionnée : 800 m rapide / 400 m récupération (répéter 8 fois)
                              - Une sortie longue : 15-18 km à rythme modéré
                              - Une séance de récupération : 40 min en course lente
                    
                            Semaine 5-8 :
                              - Une sortie tempo : 12-15 km à 85-90% de votre effort max
                              - Une séance fractionnée : 1 000 m rapide / 500 m récupération (répéter 6 fois)
                              - Une sortie longue : 18-21 km à rythme modéré
                              - Une séance de récupération : 45 min en course lente
                    
                            Semaine 9-10 :
                              - Une sortie tempo : 15 km à 90% de votre effort max
                              - Une sortie longue : 21 km à rythme modéré
                              - Testez-vous : Courrez un semi-marathon (21 km) à votre meilleur rythme. 🎉
                            """
                        )

                        if record_choisi == "21 km (semi-marathon)" and niveau_personne == 3:
                            tk.messagebox.showinfo(
                                "Programme Running Expert - Semi-marathon",
                                """
                                Semi-marathon : Objectif - Performances d'élite en 12 semaines
                                Fréquence : 6 à 7 fois par semaine
                                Focus : Seuil, endurance et puissance.
                        
                                Semaine 1-4 :
                                  - Séance 1 : 6x2 000 m à allure semi, récupération 2 min
                                  - Séance 2 : 16 km à allure seuil (80-85% VMA)
                                  - Séance 3 : 22-25 km en endurance fondamentale
                                  - Séance 4 : 10x1 000 m à 90-95% VMA, récupération 2 min
                                  - Séance 5 : 60 min en course lente ou repos actif
                                  - Séance 6 : 25 km en endurance lente
                        
                                Semaine 5-8 :
                                  - Séance 1 : 7x1 500 m à 90-95% VMA, récupération 2 min
                                  - Séance 2 : 18 km à allure seuil
                                  - Séance 3 : 28 km en endurance fondamentale
                                  - Séance 4 : 12x500 m à 100% VMA
                                  - Séance 5 : 75 min en course lente
                        
                                Semaine 9-12 :
                                  - Réduction progressive.
                                  - Testez-vous : Courez un semi-marathon à votre meilleur rythme. 🎉
                                """
                            )

                    if record_choisi == "42 km (marathon)" and niveau_personne == 0:
                        tk.messagebox.showinfo(
                            "Programme Running Débutant - Marathon",
                            """
                            Marathon : Objectif - Courir 42 km en 16 semaines
                            Fréquence : 4 à 5 fois par semaine
                            Focus : Endurance, sorties longues, et préparation mentale.
                    
                            Semaine 1-4 :
                              3 à 4 sorties par semaine
                              - Exemple : 8 à 10 km en course lente
                              - Une sortie longue : 12 à 15 km
                              - Travaillez la régularité et la technique.
                    
                            Semaine 5-8 :
                              4 sorties par semaine
                              - Une sortie longue : 18 à 22 km en course lente
                              - Une séance de vitesse : 3 min rapide / 2 min lent (répéter 6 fois)
                              - Une sortie modérée : 12 à 15 km.
                    
                            Semaine 9-12 :
                              4 à 5 sorties par semaine
                              - Une sortie longue : 25 à 30 km en course lente
                              - Séance fractionnée : 4 min rapide / 2 min lent (répéter 6 fois)
                              - Une sortie modérée : 15 à 18 km.
                    
                            Semaine 13-15 :
                              5 sorties par semaine
                              - Une sortie longue : 32 à 35 km en course lente
                              - Séance de vitesse : 5 min rapide / 2 min lent (répéter 5 fois)
                              - Une sortie modérée : 15 à 20 km.
                    
                            Semaine 16 :
                              3 sorties légères (réduction progressive)
                              - Exemple : 8 à 10 km en course lente
                              - Jour J : Marathon (42 km). 🎉
                            """
                        )

                    if record_choisi == "42 km (marathon)" and niveau_personne == 1:
                        tk.messagebox.showinfo(
                            "Programme Running Intermédiaire - Marathon",
                            """
                            Marathon : Objectif - Courir un marathon en 14 semaines
                            Fréquence : 4 à 5 fois par semaine
                            Focus : Endurance, sorties longues, et gestion de l'effort.
                    
                            Semaine 1-4 :
                              - Une sortie modérée : 10 à 12 km
                              - Une séance fractionnée : 2 min rapide / 1 min lent (répéter 8 fois)
                              - Une sortie longue : 18 à 20 km en course lente
                              - Une séance de récupération : 35 min en course lente
                    
                            Semaine 5-8 :
                              - Une sortie modérée : 12 à 15 km
                              - Une séance fractionnée : 3 min rapide / 2 min lent (répéter 6 fois)
                              - Une sortie longue : 22 à 25 km à rythme modéré
                              - Une séance de récupération : 40-45 min en course lente
                    
                            Semaine 9-12 :
                              - Une sortie modérée : 15 à 18 km
                              - Une séance fractionnée : 4 min rapide / 2 min lent (répéter 5 fois)
                              - Une sortie longue : 28 à 32 km en course lente
                    
                            Semaine 13-14 :
                              - Réduction progressive : 2 à 3 sorties par semaine
                              - Exemple : 8 à 10 km en course lente
                              - Jour J : Marathon (42 km). 🎉
                            """
                        )

                    if record_choisi == "42 km (marathon)" and niveau_personne == 2:
                        tk.messagebox.showinfo(
                            "Programme Running Avancé - Marathon",
                            """
                            Marathon : Objectif - Maximiser votre performance en 14 semaines
                            Fréquence : 5 à 6 fois par semaine
                            Focus : Endurance, sorties longues, et gestion des allures.
                    
                            Semaine 1-4 :
                              - Une sortie tempo : 12-15 km à 85% de votre effort max
                              - Une séance fractionnée : 1 000 m rapide / 500 m récupération (répéter 6 fois)
                              - Une sortie longue : 20-25 km à rythme modéré
                              - Une séance de récupération : 45 min en course lente
                    
                            Semaine 5-8 :
                              - Une sortie tempo : 15-18 km à 85-90% de votre effort max
                              - Une séance fractionnée : 1 500 m rapide / 800 m récupération (répéter 5 fois)
                              - Une sortie longue : 25-30 km à rythme modéré
                              - Une séance de récupération : 50 min en course lente
                    
                            Semaine 9-12 :
                              - Une sortie tempo : 18-20 km à 90% de votre effort max
                              - Une séance fractionnée : 2 000 m rapide / 1 000 m récupération (répéter 4 fois)
                              - Une sortie longue : 30-35 km en course lente
                    
                            Semaine 13-14 :
                              - Réduction progressive : 2 à 3 sorties par semaine
                              - Exemple : 10-15 km en course lente
                              - Jour J : Marathon (42 km). 🎉
                            """
                        )

                    if record_choisi == "42 km (marathon)" and niveau_personne == 3:
                        tk.messagebox.showinfo(
                            "Programme Running Expert - Marathon",
                            """
                            Marathon : Objectif - Record d'élite en 16 semaines
                            Fréquence : 6 à 7 fois par semaine
                            Focus : Endurance longue, vitesse et récupération.
                    
                            Semaine 1-6 :
                              - Séance 1 : 6x3 000 m à allure marathon, récupération 3 min
                              - Séance 2 : 20 km à allure seuil (80-85% VMA)
                              - Séance 3 : 30-35 km en endurance fondamentale
                              - Séance 4 : 10x1 000 m à 90-95% VMA
                              - Séance 5 : 90 min en course lente
                              - Séance 6 : 35 km en endurance lente
                    
                            Semaine 7-12 :
                              - Séance 1 : 5x5 000 m à allure marathon, récupération 5 min
                              - Séance 2 : 25 km à allure seuil
                              - Séance 3 : 36-38 km en endurance fondamentale
                              - Séance 4 : 12x400 m à 100% VMA
                              - Séance 5 : 120 min en course lente
                    
                            Semaine 13-16 :
                              - Réduction progressive (tapering) : 60-70% du volume
                              - Testez-vous : Courez un marathon à votre meilleur rythme. 🎉
                            """
                        )


                else:
                    tk.messagebox.showerror("Erreur", "Votre prénom n'a pas été trouvé dans la liste des utilisateurs.")
                popup.destroy()
            else:
                tk.messagebox.showerror("Erreur", "Veuillez sélectionner une distance.")
    
        popup = tk.Toplevel(root)
        popup.title("Choix du record à améliorer")
        popup.geometry("400x200")
        popup.configure(bg="#f4f4f4")
        
        label_question = tk.Label(popup, text="Quel record souhaitez-vous améliorer ?", font=("Arial", 14), bg="#f4f4f4")
        label_question.pack(pady=10)
        
        distances = ["5 km", "10 km", "21 km (semi-marathon)", "42 km (marathon)"]
        combo_record = ttk.Combobox(popup, values=distances, font=("Arial", 12), state="readonly")
        combo_record.pack(pady=10)
        
        ttk.Button(popup, text="Valider", command=valider_record).pack(pady=10)


    def ouvrir_fenetre_courses():
        fenetre_courses = tk.Toplevel(root)
        fenetre_courses.title("Courses de " + prenom_utilisateur)
        fenetre_courses.geometry("1200x800")
    
        # Créer un canvas avec une scrollbar pour la zone défilante
        canvas = tk.Canvas(fenetre_courses)
        scrollbar = tk.Scrollbar(fenetre_courses, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
    
        # Créer un frame dans le canvas pour contenir les widgets
        frame_courses = tk.Frame(canvas)
    
        # Mettre la frame dans le canvas
        canvas.create_window((0, 0), window=frame_courses, anchor="nw")
        canvas.pack(side="left", expand=True, fill="both", padx=20, pady=20)
        scrollbar.pack(side="right", fill="y")
    
        # Ajouter les en-têtes de colonnes
        headers = ["Nom de la course", "Distance (km)", "Vitesse Moyenne (km/h)", "Durée"]
        for col, header in enumerate(headers):
            tk.Label(frame_courses, text=header, font=("Arial", 14, "bold"), borderwidth=1, relief="solid").grid(row=0, column=col, padx=5, pady=5)
    
        # Filtrer les courses pour l'utilisateur
        user_courses = data_cleaned_complet[data_cleaned_complet["Nom"] == prenom_utilisateur]
    
        date = "Date de l'activité"
    
        # Ajouter les données des courses dans la grille
        for row, (_, course) in enumerate(user_courses.iterrows(), start=1):
            tk.Label(frame_courses, text=course["ID de l'activité"], font=("Arial", 12), borderwidth=1, relief="solid").grid(row=row, column=0, padx=5, pady=5)
            tk.Label(frame_courses, text=f"{course['Distance']:.2f}", font=("Arial", 12), borderwidth=1, relief="solid").grid(row=row, column=1, padx=5, pady=5)
            tk.Label(frame_courses, text=f"{course['Vitesse moyenne']:.2f}", font=("Arial", 12), borderwidth=1, relief="solid").grid(row=row, column=2, padx=5, pady=5)
            tk.Label(frame_courses, text=f"{course['Temps en minutes']:.2f}", font=("Arial", 12), borderwidth=1, relief="solid").grid(row=row, column=3, padx=5, pady=5)
            tk.Label(frame_courses, text=f"{course[date]}", font=("Arial", 12), borderwidth=1, relief="solid").grid(row=row, column=4, padx=5, pady=5)

            # Ajouter un bouton pour ouvrir le fichier HTML
            bouton_ouvrir = tk.Button(
                frame_courses,
                text="Voir",
                font=("Arial", 12),
                command=lambda fichier=course['Nom du fichier']: ouvrir_gpx(fichier)
            )
            bouton_ouvrir.grid(row=row, column=6, padx=5, pady=5)
    
        # Mettre à jour la taille du canvas pour inclure tous les éléments
        frame_courses.update_idletasks()
        canvas.config(scrollregion=canvas.bbox("all"))

    
    def ouvrir_gpx(nom_fichier):
        file_path = f"/Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/Data_dossier/Data_{prenom_utilisateur}/{nom_fichier}"

        with open(file_path, 'r') as gpx_file:
            gpx = gpxpy.parse(gpx_file)

        coordinates = []
        for track in gpx.tracks:
            for segment in track.segments:
                for point in segment.points:
                    coordinates.append((point.latitude, point.longitude))

        if coordinates:
            map_center = coordinates[0]
        else:
            tk.messagebox.showerror("Erreur", "Aucune coordonnée trouvée dans le fichier GPX.")
            return

        map_gpx = folium.Map(location=map_center, zoom_start=13)
        folium.PolyLine(coordinates, color="blue", weight=2.5, opacity=1).add_to(map_gpx)

        output_file = "map_gpx.html"
        map_gpx.save(output_file)
        webbrowser.open(f"file:///Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/programme python/{output_file}")


    def ouvrir_fenetre_1():
        nom_gpx = tableau_proximite.loc[tableau_proximite['Personne'] == prenom_utilisateur, 'GPX']
        nom_pers_proche = tableau_proximite.loc[tableau_proximite['Personne'] == prenom_utilisateur, 'Personne la plus proche']

        file_path = f"/Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/Data_dossier/Data_{nom_pers_proche.iloc[0]}/activities/{nom_gpx.iloc[0]}"

        with open(file_path, 'r') as gpx_file:
            gpx = gpxpy.parse(gpx_file)

        coordinates = []
        for track in gpx.tracks:
            for segment in track.segments:
                for point in segment.points:
                    coordinates.append((point.latitude, point.longitude))

        if coordinates:
            map_center = coordinates[0]
        else:
            tk.messagebox.showerror("Erreur", "Aucune coordonnée trouvée dans le fichier GPX.")
            return

        map_gpx = folium.Map(location=map_center, zoom_start=13)
        folium.PolyLine(coordinates, color="blue", weight=2.5, opacity=1).add_to(map_gpx)

        output_file = "map_gpx.html"
        map_gpx.save(output_file)
        webbrowser.open(f"file:///Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/programme python/{output_file}")

    # def ouvrir_fenetre_2():
    #     fenetre_2 = tk.Toplevel(root)
    #     fenetre_2.title("Fenêtre 2")
    #     fenetre_2.geometry("1200x800")
    #     fenetre_2.configure(bg="#f4f4f4")

    #     image_path = images_utilisateurs.get(prenom_utilisateur, "")
    #     if image_path:
    #         image = Image.open(image_path).resize((400, 400), Image.Resampling.LANCZOS)
    #         tk_image = ImageTk.PhotoImage(image)
    #         fenetre_2.tk_image = tk_image
    #         label_image = tk.Label(fenetre_2, image=tk_image, bg="#f4f4f4")
    #         label_image.pack(pady=20)

    #     ttk.Label(fenetre_2, text=f"Bonjour {prenom_utilisateur}, bienvenue !", font=("Arial", 20)).pack(pady=20)
    #     ttk.Button(fenetre_2, text="Fermer", command=fenetre_2.destroy).pack(pady=20)

    frame_images = tk.Frame(root, bg="#f4f4f4")
    frame_images.pack(side=tk.TOP, pady=20)

    image_path1 = f"/Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/RESULTATS/resultats_{prenom_utilisateur}/Evolution_Pace_Modifiée_{prenom_utilisateur}_ARIMA.png"
    if image_path1:
        image1 = Image.open(image_path1).resize((650, 520), Image.Resampling.LANCZOS)
        tk_image1 = ImageTk.PhotoImage(image1)
        label_image1 = tk.Label(frame_images, image=tk_image1, bg="#f4f4f4")
        label_image1.grid(row=0, column=0, padx=20)

    image_path2 = f"/Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/RESULTATS/resultats_{prenom_utilisateur}/Evolution_Records_Linear_{prenom_utilisateur}.png"
    if image_path2:
        image2 = Image.open(image_path2).resize((650, 520), Image.Resampling.LANCZOS)
        tk_image2 = ImageTk.PhotoImage(image2)
        label_image2 = tk.Label(frame_images, image=tk_image2, bg="#f4f4f4")
        label_image2.grid(row=0, column=1, padx=20)

    coordinates = data_without_total.iloc[0:N, :].to_numpy()
    for i, name in enumerate(data_without_total.index.to_list()):
        if name == prenom_utilisateur:
            C1 = coordinates[i, 0]*100
            C2 = coordinates[i, 1]*100
            C3 = coordinates[i, 2]*100
            C4 = coordinates[i, 3]*100
            
    label_texte = tk.Label(root, text=f"Bienvenue {prenom_utilisateur} !\n\n"
                                     f"Débutant : {C1:.1f}%\n"
                                     f"Intermédiaire : {C3:.1f}%\n"
                                     f"Avancé : {C2:.1f}%\n"
                                     f"Expert : {C4:.1f}%",
                           font=("Arial", 18), bg="#f4f4f4", fg="blue")
    label_texte.pack(pady=20)

    frame_boutons = tk.Frame(root, bg="#f4f4f4")
    frame_boutons.pack(side=tk.BOTTOM, pady=20)

    ttk.Button(frame_boutons, text="Course recommandée", command=ouvrir_fenetre_1).grid(row=0, column=0, padx=20)
    #ttk.Button(frame_boutons, text="Ouvrir Fenêtre 2", command=ouvrir_fenetre_2).grid(row=0, column=1, padx=20)
    ttk.Button(frame_boutons, text="Voir Mes Courses", command=ouvrir_fenetre_courses).grid(row=0, column=3, padx=20)
    ttk.Button(frame_boutons, text="Choisir un record à améliorer", command=demander_record).grid(row=0, column=4, padx=20)


    def afficher_graphe_3D():
        """Affiche un graphe 3D avec les clusters K-means."""
        # Calcul des clusters et des distances
        data = pd.read_csv('/Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/resultats t-SNE/Data_complet_ap_tSNE.csv')
        data_without_total = pd.read_csv('/Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/data_without_total1.csv')
    
        with open("/Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/noms_lignes.txt", "r") as f:
            noms_lignes = [line.strip() for line in f.readlines()]
    
        data_without_total.index = noms_lignes
        
        noms_distincts = list(data["Nom"].unique())
        N = len(noms_distincts)
    
        coordinates = data_without_total.iloc[0:N, :].to_numpy()
        distances = squareform(pdist(coordinates, metric='euclidean'))
        kmeans = KMeans(n_clusters=4, random_state=42)
        kmeans.fit(coordinates)
        data_without_total['Cluster'] = kmeans.labels_
    
        # Créer une nouvelle fenêtre
        fenetre_graphe = tk.Toplevel(root)
        fenetre_graphe.title("Graphe 3D")
        fenetre_graphe.geometry("1200x800")
    
        # Générer le graphe 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(
            data_without_total.iloc[0:N, 0], 
            data_without_total.iloc[0:N, 1], 
            data_without_total.iloc[0:N, 2], 
            c=kmeans.labels_, 
            cmap='viridis'
        )
        
        ax.set_title('K-means Clustering (3D)')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
    
        # Ajouter les noms des personnes
        for i, name in enumerate(data_without_total.index.to_list()):
            if name == prenom_utilisateur:
                ax.text(coordinates[i, 0], coordinates[i, 1], coordinates[i, 2], name, fontsize=15)
            else:
                ax.text(coordinates[i, 0], coordinates[i, 1], coordinates[i, 2], name, fontsize=8)
    
        # Intégrer le graphe dans Tkinter
        canvas = FigureCanvasTkAgg(fig, master=fenetre_graphe)
        canvas.draw()
        canvas.get_tk_widget().pack(expand=True, fill="both")
    
        # Ajouter une barre d'outils de navigation
        toolbar = NavigationToolbar2Tk(canvas, fenetre_graphe)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        
    ttk.Button(frame_boutons, text="Afficher Graphe 3D", command=afficher_graphe_3D).grid(row=0, column=2, padx=20)
    # Ajouter le bouton dans l'interface principale
    # bouton_3D = tk.Button(frame_boutons, text="Afficher Graphe 3D", command=afficher_graphe_3D, font=("Arial", 16))
    # bouton_3D.grid(row=0, column=2, padx=40)
    
    # Lancer la boucle principale
    root.mainloop()

def demander_prenom():
    def valider_prenom():
        global prenom_utilisateur
        prenom_utilisateur = entry_prenom.get()
        popup.destroy()
        afficher_interface_principale()

    popup = tk.Tk()
    popup.title("Bienvenue")
    popup.geometry("400x200")
    popup.configure(bg="#f4f4f4")

    label_question = tk.Label(popup, text="Comment vous appelez-vous ?", font=("Arial", 14), bg="#f4f4f4")
    label_question.pack(pady=10)

    entry_prenom = ttk.Entry(popup, font=("Arial", 14))
    entry_prenom.pack(pady=10)

    ttk.Button(popup, text="Valider", command=valider_prenom).pack(pady=10)

    popup.mainloop()

demander_prenom()
