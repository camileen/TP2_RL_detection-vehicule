import os
import cv2
import numpy as np
import random
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from time import time

DATASET_PATH = "/mnt/c/Users/byoub/Downloads/data/"
NB_VEHICLES = 10
NB_NON_VEHICLES = 10

# Charger le dataset depuis Kaggle
def load_dataset(folder_path, label):
  dataset = []
  list_files = os.listdir(folder_path)
  for file in list_files[:NB_VEHICLES if label == 1 else NB_NON_VEHICLES]:
    img_path = os.path.join(folder_path, file)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # Convertir en niveaux de gris
    img = cv2.resize(img, (64, 64)) # Redimensionner
    features = extract_hog_features(img) # Extraire les caractéristiques HOG
    dataset.append({'features': features, 'label': label})
  return dataset

# Extraction des descripteurs HOG
def extract_hog_features(image): 
  features, _ = hog(image, 
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    orientations=9,
    visualize=True,
    block_norm='L2-Hys')
  return features



# Définition de l'environnement
class Environment:
  def __init__(self, dataset):
    self.dataset = dataset
    self.current_index = 0

  def reset(self):
    self.current_index = 0
    return self.dataset[self.current_index]['features']

  def step(self, action):
    reward = 1 if action == self.dataset[self.current_index]['label'] else -1
    self.current_index += 1
    done = self.current_index >= len(self.dataset)
    next_state = self.dataset[self.current_index]['features'] if not done else None
    return next_state, reward, done

# Agent Q-Learning
class QLearningAgent:
  def __init__(self, action_space):
    self.q_table = {}
    self.action_space = action_space
    self.learning_rate = 0.1
    self.gamma = 0.9

  def choose_action(self, state):
    if state not in self.q_table:
      self.q_table[state] = np.zeros(self.action_space)
    return np.argmax(self.q_table[state]) if random.random() > 0.1 else random.randint(0, self.action_space - 1)

  def update_q_value(self, state, action, reward, next_state):
    if next_state is None:
      target = reward
    else:
      target = reward + self.gamma * np.max(self.q_table.get(next_state, np.zeros(self.action_space)))
      self.q_table[state][action] += self.learning_rate * (target - self.q_table[state][action])

def simulation(dataset):
  env = Environment(dataset)
  agent = QLearningAgent(action_space=2)
  print("Simulation en cours...")
  start = time()
  for episode in range(10):
    state = env.reset()
    done = False
    while not done:
      action = agent.choose_action(state)
      next_state, reward, done = env.step(action)
      agent.update_q_value(state, action, reward, next_state)
      state = next_state
  end = time()
  print(f"==> FIN : Durée de la simulation: {end - start:.2f} secondes")
  return agent
  


if __name__ == "__main__":
  # Charger les images des deux classes
  print("Chargement des images...")
  vehicle_images = load_dataset(f"{DATASET_PATH}vehicles", label=1)
  non_vehicle_images = load_dataset(f"{DATASET_PATH}non-vehicles", label=0)
  dataset = vehicle_images + non_vehicle_images

  # Normaliser les caractéristiques
  scaler = StandardScaler()
  features_matrix = np.array([data['features'] for data in dataset])
  scaled_features = scaler.fit_transform(features_matrix)

  # Mettre à jour le dataset avec les caractéristiques normalisées
  for i, data in enumerate(dataset): dataset[i]['features'] = tuple(scaled_features[i]) # Convertir en tuple pour l'utiliser comme clé dans la Q-Table

  agent = simulation(dataset)

  # Afficher la Q-Table
  print("Q-Table (partielle):", list(agent.q_table.items())[:5])