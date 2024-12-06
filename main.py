!pip install wget

# importing the nessesory library
import numpy as np
import re
from sklearn.metrics.pairwise import euclidean_distances
import random
import wget
import zipfile

url='https://nlp.stanford.edu/data/glove.840B.300d.zip'
file='glovedata.zip'
wget.download(url, file)
print('Downloaded')

with zipfile.ZipFile('/content/glovedata.zip', 'r') as zip_ref:
    zip_ref.extractall('glovefile.txt')
print('Extracted')

File='/content/SMSSpamCollection'

"""# **Load Data files**"""

# Load the Glove Vectors
def loadGlovevector(file_path):
    glove_vector = {}
    with open(file_path, 'r', encoding="utf-8") as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            try:
                vector = np.array(values[1:], dtype='float32')
                if len(vector) == 300: # Ensure it's a 300-dimensional vector
                    glove_vector[word] = vector
            except ValueError:
                continue # Skip lines with non-numeric values
    return glove_vector

def load_messages_from_txt(file_path):
    messages = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip(): # Skip empty lines
                label, message = line.split(maxsplit=1)
                messages.append((label.lower(), message.strip()))
    return messages

"""# **Tokenize and compute message vector**"""

def tokenize_message(message):
  return re.findall(r'\b\w+\b',message.lower())

def compute_message_vector(message, glove_vector):
  tokens=tokenize_message(message)
  D= len(tokens)

  message_vector=np.zeros(300)
  for word in tokens:
    if word in glove_vector:
      word_vector=glove_vector[word]
      prob=(tokens.count(word)+1)/(D+2)
      message_vector+=prob*word_vector
  return message_vector

def process_messages(messages, glove_vector):
  vectors=[]
  for label,message in messages:
    message_vector=compute_message_vector(message, glove_vector)
    vectors.append((label,message_vector))
  return vectors

"""# **top 10 representive Vectors**"""

# find the top 10 representive Vectors
def find_top_representatives(vectors, label):
    category_vectors = np.array([vec for lbl, vec in vectors if lbl == label])
    mean_distances = []

    for vec in category_vectors:
        distances = euclidean_distances([vec], category_vectors)
        mean_distance = np.mean(distances)
        mean_distances.append((mean_distance, vec))

    mean_distances.sort(key=lambda x: x[0])
    top_10 = [vec for _, vec in mean_distances[:10]]
    return top_10

"""# **File Save**"""

# Save vectors to file
def save_vectors(vectors, filename):
  with open(filename, 'w') as f:
    for _, vector in vectors:
      vector_str = ' '.join(map(str, vector))
      f.write(f"{vector_str}\n")

def save_top_representatives(vectors, filename):
    with open(filename, 'w') as f:
        for label in ['ham', 'spam']:
            top_10 = find_top_representatives(vectors, label)
            for vector in top_10:
                vector_str = ' '.join(map(str, vector))
                f.write(f"{label} {vector_str}\n")

"""# **Calculate cohension Ratio**"""

def cohesion(vectors):
    pairwise_dist = euclidean_distances(vectors)
    return np.mean(pairwise_dist)

def cohesion_ratio(vectors, label):
    top_10 = find_top_representatives(vectors, label)
    category_vectors = [vec for lbl, vec in vectors if lbl == label]
    random_10 = random.sample(category_vectors, 10)

    coh_top_10 = cohesion(top_10)
    coh_random_10 = cohesion(random_10)

    return coh_top_10 / coh_random_10

def save_cohesion_ratios(vectors):
    for label in ['ham', 'spam']:
        ratio = cohesion_ratio(vectors, label)
        print(f"Cohesion Ratio for {label}: {ratio:.3f}")

"""# **Main function**"""

glove_vector=loadGlovevector('/content/glovefile.txt/glove.840B.300d.txt')

messages=load_messages_from_txt(File)

processed_vectors=process_messages(messages, glove_vector)

save_vectors(processed_vectors, 'message_vectors.txt')

save_top_representatives(processed_vectors, 'top_representatives.txt')

save_cohesion_ratios(processed_vectors)
