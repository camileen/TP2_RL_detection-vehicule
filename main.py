import os
import cv2
import numpy as np
import random
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import json
from time import time
import datetime

DATASET_PATH = "/mnt/c/Users/byoub/Downloads/data/" 
NB_VEHICLES = 8792 # 8792 au total
NB_NON_VEHICLES = 8968 # 8968 au total
NB_Q_TABLES = 10


# Charger le dataset depuis Kaggle
def load_dataset(folder_path, label):
  dataset = []
  files = os.listdir(folder_path)
  for file in files[:(NB_VEHICLES if label == 1 else NB_NON_VEHICLES)]:
    img_path = os.path.join(folder_path, file)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # Convertir en niveaux de gris
    img = cv2.resize(img, (64, 64)) # Redimensionner

    if img is None:
      print(f"[WARNING] Could not load image: {img_path}")
      continue  # Skip broken or unreadable files

    features = extract_hog_features(img) # Extraire les caractéristiques HOG
    
    if np.all(features == 0):
      print(f"[WARNING] All-zero HOG features in: {img_path}")
      #features += np.random.normal(0, 1e-4, size=features.shape) # Add small noise

    if np.any(np.isnan(features)):
      print(f"[ERROR] NaN found in HOG features: {img_path}")

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
    random.shuffle(self.dataset)  # Mélanger le dataset à chaque réinitialisation
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
  def __init__(self, action_space, learning_rate=0.1, gamma=0.9):
    self.q_table = {}
    self.action_space = action_space
    self.learning_rate = learning_rate
    self.gamma = gamma

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


def get_dataset():
  # Charger les images des deux classes
  print("LOAD IMAGES...")

  # Load labels.json
  labels_path = os.path.join(DATASET_PATH, "labels.json")
  with open(labels_path, "r") as f:
    labels = json.load(f)
  
  start_load = time()
  dataset = []
  for img_path, label in labels:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 64)) # Redimensionner

    if img is None:
      print(f"[WARNING] Could not load image: {img_path}")
      continue  # Skip broken or unreadable files

    features = extract_hog_features(img) # Extraire les caractéristiques HOG

    if np.all(features == 0):
      print(f"[WARNING] All-zero HOG features in: {img_path}")
      #features += np.random.normal(0, 1e-4, size=features.shape) # Add small noise

    if np.any(np.isnan(features)):
      print(f"[ERROR] NaN found in HOG features: {img_path}")
    
    dataset.append({'features': features, 'label': label})
  end_load = time()
  print("=> LOADING TIME: ", end_load - start_load, "seconds")
  print(f"TOTAL: Loaded {len(dataset)} samples of vehicles and non-vehicles")
  print("====== Loading done ======\n")
  

  # Normaliser les caractéristiques
  scaler = StandardScaler()
  features_matrix = np.array([data['features'] for data in dataset])
  scaled_features = scaler.fit_transform(features_matrix)

  # Mettre à jour le dataset avec les caractéristiques normalisées
  for i, data in enumerate(dataset): 
    dataset[i]['features'] = tuple(np.round(scaled_features[i], 1)) # Convertir en tuple pour l'utiliser comme clé dans la Q-Table
        
  return dataset


def save_results(learning_rate, nb_episodes, gamma, results, type):
  # Sauvegarder les métriques de performance
  dir = f"results/{type + 's' if type == 'q-table' else type}/"
  filename = f"{type}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_{learning_rate}-{nb_episodes}-{gamma}.json"
  path = os.path.join(dir, filename)

  res = None
  if type == "q-table":
    # Convert keys to strings because JSON keys must be strings
    # res = {str(state): list(values) for state, values in results.items()}
    res = {}
    i = 0
    for state, values in results.items():
      if i < NB_Q_TABLES:
        res[str(state)] = list(values)
        i += 1
      else: continue
  else:
    res = results

  with open(path, mode="w", newline="") as file:
    json.dump(res, file, indent=4)
  print(f"=> OUTPUT RESULT: {path}")


def simulation(dataset, learning_rate, nb_episodes, gamma):
  env = Environment(dataset)
  agent = QLearningAgent(action_space=2, learning_rate=learning_rate, gamma=gamma)

  all_metrics = []
  all_rewards = []

  print("START SIMULATION...")
  start = time()
  for episode in range(nb_episodes):
    y_true, y_prediction, rewards = [], [], []
    state = env.reset()
    done = False
    while not done:
      action = agent.choose_action(state)
      next_state, reward, done = env.step(action)

      y_prediction.append(action)
      y_true.append(env.dataset[env.current_index - 1]['label'])
      rewards.append(reward)

      agent.update_q_value(state, action, reward, next_state)
      state = next_state
    
    metrics = {
      "episode": episode,
      "accuracy": accuracy_score(y_true, y_prediction),
      "precision": precision_score(y_true, y_prediction, zero_division=0),
      "recall": recall_score(y_true, y_prediction, zero_division=0),
      "f1_score": f1_score(y_true, y_prediction, zero_division=0)
    }
    all_metrics.append(metrics)
    all_rewards.append(sum(rewards))
    print(f"Episode {episode + 1}/{nb_episodes} - Accuracy: {metrics['accuracy']:.2f} - Precision: {metrics['precision']:.2f} - Recall: {metrics['recall']:.2f} - F1 Score: {metrics['f1_score']:.2f}")

  end = time()
  print("=> FINISHED IN: ", end - start, "seconds")
  return all_metrics, agent.q_table, all_rewards


if __name__ == "__main__":

  dataset = get_dataset()
  
  # Simulation
  lr = 0.1
  nb = 10
  gamma = 0.1

  all_metrics, agent_q_table, all_rewards = simulation(dataset=dataset, learning_rate=lr, nb_episodes=nb, gamma=gamma)

  # Affiche la Q-Table
  # print("Q-Table (partielle):", list(agent_q_table.items())[:2])

  # Sauvegarde les métriques de performance et la Q-Table
  save_results(learning_rate=lr, nb_episodes=nb, gamma=gamma, results=all_metrics, type="metrics")
  save_results(learning_rate=lr, nb_episodes=nb, gamma=gamma, results=agent_q_table, type="q-table")
  save_results(learning_rate=lr, nb_episodes=nb, gamma=gamma, results=all_rewards, type="rewards")

  print("END")
