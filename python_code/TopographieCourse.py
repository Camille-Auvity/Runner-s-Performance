## Projet Industriel - Visualiser un Fichier GPX

import gpxpy
import gpxpy.gpx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from math import sin, cos, sqrt, atan2, radians
import numpy as np
import pandas as pd
import os
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# Créer une fenêtre Tkinter
root = tk.Tk()
root.title("Graphique dans une autre fenêtre")

gpx_file = open(os.path.join("/Users/camilleauvity/Downloads", "10kmChampsElysees2025 (3).gpx"), "r")
gpx = gpxpy.parse(gpx_file)

def getLocationFromCoords (geolocator, latitude, longitude):
    coord = str(latitude)+ "," + str(longitude)
    location = geolocator.reverse(coord)
    return location.raw

def CoordsFromLocation ( geolocator, cityName ):
    coords = geolocator.geocode(cityName)
    return coords

def getDistanceBetweenCoords (lat1, long1, lat2, long2):
    earthRadius = 6371
    c = 0
    
    if all ((lat1, long1, lat2, long2)):
        latFrom = radians(lat1)
        lonFrom = radians(long1)
        latTo = radians(lat2)
        lonTo = radians(long2)
        
        dlon = lonTo - lonFrom
        dlat = latTo - latFrom
        
        a = sin(dlat / 2 )**2 + cos(latFrom) * cos(latTo) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    return earthRadius * c


trackDf = []        #final dataframe initialization
segUnit = 0.05         #segment unit, here 50 m
km = 0              #cumulated km initialization

lastLat = None      #previously read latitude, None at the moment
lastLon = None      #previously read longitude, None at the moment

#We iterate on gpx structure then ...
for track in gpx.tracks:
    for segment in track.segments:
        for point in segment.points:

            # ... we compute the distance between current waypoint and last one
            distFromLastStep = getDistanceBetweenCoords(lastLat, lastLon, point.latitude, point.longitude)
            km = km + distFromLastStep  #cumulated km is updated
            
            # ... current segment determination
            segment = km // segUnit
            
            trackDf.append([segment, km, point.latitude, point.longitude, point.elevation])
            
            # ... current coords become previous ones
            lastLat = point.latitude
            lastLon = point.longitude


trackDf = pd.DataFrame(trackDf)  
trackDf.columns = ['segment', 'km', 'latitude', 'longitude', 'elevation']

#---------------------------
slopesDf = []                      # dataframe initialization

# For each segment we calculated previously ...
for segment in trackDf['segment'].unique():
    
    #get first step of current segment then its elevation
    firstSt = trackDf.loc[trackDf[trackDf['segment'].eq(segment)]['km'].idxmin()]
    fromElv = firstSt['elevation'] 
    
    #get last step of current segment then its elevation
    lastSt = trackDf.loc[trackDf[trackDf['segment'].eq(segment)]['km'].idxmax()]
    toElv = lastSt['elevation'] 
 
    #determine current segment lenght (in km)
    segmentLength = trackDf[trackDf['segment'].eq(segment)]['km'].max() - trackDf[trackDf['segment'].eq(segment)]['km'].min()
    
    #calculate slope of current segment(in %)
    segmentSlope = (toElv - fromElv) / (segmentLength * 1000) * 100
    
    slopesDf.append([segment, segmentSlope])
    
    
slopesDf = pd.DataFrame(slopesDf)  
slopesDf.columns = ['segment', 'slope']

trackDf = pd.merge(trackDf, slopesDf, on='segment')
trackDf.head()

#-----------------------------

# slopes rules divisions
slopesTable = [lambda x: x < 2, 
               lambda x: (x >= 2) & (x < 4),
               lambda x: (x >= 4) & (x < 5),
               lambda x: (x >= 5) & (x < 8),
               lambda x: x >= 8,]

# slopes associated colors
slopesColor = ['palegreen', 'yellow', 'orange', 'orangered', 'maroon']

# slopes legend
slopesDescr = ['inf 2%', '2 ~ 4%', '4 ~ 5%', '5 ~ 8%', 'sup 8%']

#-----------------------------

def smoothProfile(signal,L=10):
    res = np.copy(signal) 
    for i in range (1,len(signal)-1): 
        L_g = min(i,L) 
        L_d = min(len(signal)-i-1,L) 
        Li=min(L_g,L_d)
        res[i]=np.sum(signal[i-Li:i+Li+1])/(2*Li+1)
    return res

#-----------------------------


legend = []                                # legend initialization
style = dict(size=10, color='gray')        # style used for annotations

# We first define the canvas, then axis (right and top ones will be hidden)
fig = plt.Figure()
ax = fig.add_subplot(111)
fig, ax = plt.subplots(figsize=(12,4))
plt.xlabel("Kilometers")
plt.ylabel("Elevation")
ax.spines[['right', 'top']].set_visible(False)


# Rendering of backhround gray curve (for a 3D effect)
# This curve is shifted 1km further and 30m above
plt.fill_between(trackDf['km']+0.2, smoothProfile(trackDf['elevation'])+1, color='gray', zorder=0)

# For each slope category we defined, profile is filled and legend is updated
for i in range(0, len(slopesTable)):
    plt.fill_between(trackDf['km'], smoothProfile(trackDf['elevation']), where = (slopesTable[i])(trackDf['slope']), color=slopesColor[i], zorder=1)
    legend.append(mpatches.Patch(color=slopesColor[i], label=slopesDescr[i]))

# We want annotations of place names to be over profile 
annotationsAnchor = trackDf['elevation'].max() * 1.1;
    
# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put the legend to the right of the current axis
ax.legend(handles=legend, loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()

# Intégrer la figure Matplotlib dans Tkinter
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack(expand=True, fill="both")  # Remplir tout l'espace

# Ajouter une barre d'outils de navigation (avec zoom et pan)
toolbar = NavigationToolbar2Tk(canvas, root)
toolbar.update()
toolbar.pack(side=tk.BOTTOM, fill=tk.X)  # Placer la barre d'outils en bas de la fenêtre


# Lancer l'application Tkinter
root.mainloop()



