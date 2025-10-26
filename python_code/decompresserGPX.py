import os
import gzip
import shutil
from fitparse import FitFile
from xml.etree.ElementTree import Element, SubElement, tostring
import xml.dom.minidom

# Charger les noms des coureurs depuis le fichier fourni
def load_runner_names(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file.readlines() if line.strip()]

# Fonction pour décompresser les fichiers .gz
def decompress_gz(input_path, output_path):
    with gzip.open(input_path, 'rb') as gz_file:
        with open(output_path, 'wb') as out_file:
            shutil.copyfileobj(gz_file, out_file)
    print(f"Décompressé : {input_path} -> {output_path}")

# Fonction pour convertir un fichier .fit en .gpx
# def fit_to_gpx(fit_file_path, gpx_file_path):
#     fitfile = FitFile(fit_file_path)
#     gpx = Element('gpx', attrib={'xmlns': 'http://www.topografix.com/GPX/1/1', 'creator': 'FIT-to-GPX', 'version': '1.1'})
#     trk = SubElement(gpx, 'trk')
#     trkseg = SubElement(trk, 'trkseg')
#     for record in fitfile.get_messages('record'):
#         lat, lon, ele, time = None, None, None, None
#         for data in record:
#             if data.name == 'position_lat':
#                 lat = data.value / ((2 ** 31) / 180.0)
#             elif data.name == 'position_long':
#                 lon = data.value / ((2 ** 31) / 180.0)
#             elif data.name == 'altitude':
#                 ele = data.value
#             elif data.name == 'timestamp':
#                 time = data.value.isoformat()
#         if lat and lon:
#             trkpt = SubElement(trkseg, 'trkpt', attrib={'lat': str(lat), 'lon': str(lon)})
#             if ele:
#                 SubElement(trkpt, 'ele').text = str(ele)
#             if time:
#                 SubElement(trkpt, 'time').text = time
#     gpx_str = xml.dom.minidom.parseString(tostring(gpx)).toprettyxml(indent="  ")
#     with open(gpx_file_path, 'w') as gpx_file:
#         gpx_file.write(gpx_str)
#     print(f"Converti FIT -> GPX : {fit_file_path} -> {gpx_file_path}")

import fitdecode

def fit_to_gpx_fitdecode(fit_file_path, gpx_file_path):
    try:
        with fitdecode.FitReader(fit_file_path) as fitfile:
            gpx = Element('gpx', attrib={'xmlns': 'http://www.topografix.com/GPX/1/1', 'creator': 'FIT-to-GPX', 'version': '1.1'})
            trk = SubElement(gpx, 'trk')
            trkseg = SubElement(trk, 'trkseg')
            
            for frame in fitfile:
                if isinstance(frame, fitdecode.FitDataMessage) and frame.name == "record":
                    lat = frame.get_value('position_lat')
                    lon = frame.get_value('position_long')
                    ele = frame.get_value('altitude')
                    time = frame.get_value('timestamp')
                    
                    if lat and lon:
                        trkpt = SubElement(trkseg, 'trkpt', attrib={
                            'lat': str(lat / ((2 ** 31) / 180.0)),
                            'lon': str(lon / ((2 ** 31) / 180.0))
                        })
                        if ele:
                            SubElement(trkpt, 'ele').text = str(ele)
                        if time:
                            SubElement(trkpt, 'time').text = time.isoformat()
            
            gpx_str = xml.dom.minidom.parseString(tostring(gpx)).toprettyxml(indent="  ")
            with open(gpx_file_path, 'w') as gpx_file:
                gpx_file.write(gpx_str)
            print(f"Converti FIT -> GPX avec fitdecode : {fit_file_path} -> {gpx_file_path}")
    except Exception as e:
        print(f"Erreur lors du traitement de {fit_file_path} avec fitdecode : {e}")


# Fonction pour copier un fichier GPX existant
def copy_gpx(input_path, output_path):
    shutil.copy(input_path, output_path)
    print(f"Copié : {input_path} -> {output_path}")

# Fonction principale pour traiter les fichiers d'un coureur
def process_runner_data(runner_name, base_input_path, output_directory):
    runner_input_path = os.path.join(base_input_path, f"Data_{runner_name}")
    runner_output_path = os.path.join(output_directory, f"Data_{runner_name}")
    os.makedirs(runner_output_path, exist_ok=True)

    for root, _, files in os.walk(runner_input_path):
        for file_name in files:
            input_path = os.path.join(root, file_name)
            if file_name.endswith('.gz'):  # Décompression des fichiers .gz
                decompressed_path = os.path.splitext(input_path)[0]
                decompress_gz(input_path, decompressed_path)
            elif file_name.endswith('.fit'):  # Conversion des fichiers .fit
                output_path = os.path.join(runner_output_path, os.path.splitext(file_name)[0] + '.gpx')
                fit_to_gpx_fitdecode(input_path, output_path)
            elif file_name.endswith('.gpx'):  # Copie des fichiers .gpx existants
                output_path = os.path.join(runner_output_path, file_name)
                copy_gpx(input_path, output_path)

# Exemple d'utilisation
if __name__ == "__main__":
    # Charger les noms des coureurs
    runner_names_file = "/Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/noms_lignes.txt"  # Chemin vers votre fichier `noms_lignes.txt`
    runner_names = load_runner_names(runner_names_file)

    # Chemins principaux
    base_input_dir = "/Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/Data_dossier"
    output_dir = "/Users/camilleauvity/Desktop/ECOLE/Projet-Industriel/GPX_Output"

    # Traiter les données pour chaque coureur
    for runner in runner_names:
        print(f"Traitement des données pour {runner}...")
        process_runner_data(runner, base_input_dir, output_dir)

    print("Traitement terminé pour tous les coureurs.")
