'''
ALGO : Modifier le dataset pour qu'il contiennne toutes les images mélangées et nommées de façon similaires'
+ faire un CSV faisant la corrspondance des images et de leur label

Début
  liste_véhicules = lire le dossier contenant les images de véhicules et récupérer les chemins de fichiers
  liste_non_véhicules = pareil pour les images de non-véhicules
  liste_des_fichiers = mélanger(liste_véhicules + liste_non_véhicules)
  Pour chaque image i:
    Déplacer dans le dossier data/
    Renommer le fichier img_i.png
  Fin Pour
Fin

'''

# NOTE : je devrais faire une copie du fichier extrait et supprimer les dossiers vehicles et non-vehicles une fois les images déplacées

import os
import random
import json

DATA_PATH = "/mnt/c/Users/byoub/Downloads/data"

NB_VEHICLES = 8792 # 8692 au total
NB_NON_VEHICLES = 8968 # 8868 au total

NB_TOTAL_IMG = NB_VEHICLES + NB_NON_VEHICLES

if __name__ == "__main__":
  vehicles_path = os.path.join(DATA_PATH, "vehicles")
  non_vehicles_path = os.path.join(DATA_PATH, "non-vehicles")

  vehicles = os.listdir(vehicles_path)
  random.shuffle(vehicles)  # Shuffle the vehicle images
  non_vehicles = os.listdir(non_vehicles_path)
  random.shuffle(non_vehicles)  # Shuffle the non-vehicle images
  all_img_files = vehicles[:NB_VEHICLES] + non_vehicles[:NB_NON_VEHICLES]

  random.shuffle(all_img_files)

  # Move and rename images
  labels = []
  for i, img_path in enumerate(all_img_files):
    new_img_path = os.path.join(DATA_PATH, f"img_{i+1}.png")
    if img_path[:5] == "extra" or img_path[:5] == "image":
      labels.append((new_img_path, 0))
      os.rename(os.path.join(non_vehicles_path, img_path), new_img_path)
    else:
      labels.append((new_img_path, 1))
      os.rename(os.path.join(vehicles_path, img_path), new_img_path)
    #os.rename(os.path.join(non_vehicles_path if img_path[:5] == "extra" or img_path[:5] == "image" else vehicles_path, img_path), new_img_path)  # Deplace AND rename the file
    print(f"Moved {img_path} to {new_img_path}")
    if i == NB_TOTAL_IMG - 1: # Useful when less images are needed
      continue

  # Save the labels to a JSON file
  labels_path = os.path.join(DATA_PATH, "labels.json")
  with open(labels_path, "w") as f:
    json.dump(labels, f)
  print(f"Labels saved to {labels_path}")

  