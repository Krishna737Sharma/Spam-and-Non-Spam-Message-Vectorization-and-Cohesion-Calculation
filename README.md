# Spam and Non-Spam Message Vectorization and Cohesion Calculation

## Objective

The goal of this assignment is to build a Python program that processes a dataset of spam and non-spam messages, converts them into fixed-dimensional dense vectors using pre-trained word vectors (GloVe), and computes cohesion scores for each category. The tasks are broken down into three parts:

1. Convert each message to a fixed-dimensional dense vector using the provided GloVe word vectors.
2. Identify the top 10 representative message vectors for each category based on the minimum within-category mean Euclidean distance.
3. Compute the cohesion ratio for each category using the top 10 vectors and random 10 vectors from the category.

## Requirements

- **Python 3.x**: Ensure Python 3.x is installed.
- **Libraries**: The following Python libraries are required:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `nltk`
  - `zipfile`
  - `requests` (if needed for downloading the dataset)
  - `re` (for regular expression-based tokenization)
  
  You can install the libraries using:
  
  ```bash
  pip install numpy pandas scikit-learn nltk requests

    GloVe Word Vectors: Download the GloVe word vectors from this link.

Instructions
1. Convert Each Message to a Fixed Dimensional Dense Vector

    Input: A dataset of spam and non-spam messages.
    Output: A file containing vectors representing each message. Each line corresponds to one message, and each vector is a space-separated list of 300 values.

Steps:

    Tokenize each message based on non-alphanumeric characters.

    Apply case folding (convert all text to lowercase).

    For each word in the message, calculate its contribution to the message vector using the formula:
    vector(D)=∑iP(wi∈D)×vector(wi)
    vector(D)=i∑​P(wi​∈D)×vector(wi​)

    where:
        P(wi∈D)=count(wi∈D)+1∣D∣+2P(wi​∈D)=∣D∣+2count(wi​∈D)+1​
        The vector for wiwi​ is obtained from the GloVe word vectors.

    Save the vector for each message to an output file, one vector per line.

2. Find the Top 10 Representative Message Vectors for Each Category

    Input: The vectors computed in the previous step.
    Output: A file containing the top 10 representative vectors for each category (spam and non-spam), ordered based on the minimum within-category mean Euclidean distance.

Steps:

    For each category (spam and non-spam), compute the mean Euclidean distance between all pairs of message vectors within the category.
    Select the 10 vectors with the smallest mean Euclidean distance from the rest of the category's vectors.
    Save the top 10 vectors for each category in a separate file, adding the category label to the first field.

3. Compute the Cohesion Ratio for Each Category

    Input: The top 10 representative vectors for each category and random 10 vectors from the same category.
    Output: The cohesion ratio for each category.

Steps:

    Calculate the pairwise Euclidean distance for the top 10 representative vectors.

    Calculate the pairwise Euclidean distance for random 10 vectors chosen from the same category.

    Compute the cohesion ratio RR as follows:
    R=coh(top 10 from category A)coh(random 10 from category A)
    R=coh(random 10 from category A)coh(top 10 from category A)​

    Print the cohesion ratio for each category.

Directory Structure

The directory structure for this assignment should look like this:

Spam_NonSpam_Vectorization/
│
├── data/
│   ├── spam_messages.txt
│   ├── non_spam_messages.txt
│   └── glove.840B.300d.zip
│
├── output/
│   ├── message_vectors.txt
│   ├── top_10_representative_vectors.txt
│   ├── cohesion_scores.txt
│
├── vectorization.py
└── README.md

    data/: Contains the input dataset and the GloVe word vectors.
    output/: Stores the results (message vectors, top 10 vectors, cohesion scores).
    vectorization.py: Python script that implements the three tasks.
    README.md: This file.

Tasks Breakdown
Task 1: Message Vectorization

    Download and extract the GloVe word vectors.
    Tokenize and process the messages.
    Compute the vectors for each message.
    Save the vectors in message_vectors.txt.

Task 2: Top 10 Representative Message Vectors

    Calculate the pairwise Euclidean distance for all vectors in each category.
    Select the top 10 vectors with the minimum within-category distance.
    Save the top 10 vectors with the category label in top_10_representative_vectors.txt.

Task 3: Cohesion Calculation

    Compute the pairwise Euclidean distance for the top 10 vectors and random 10 vectors.
    Compute the cohesion ratio.
    Save the cohesion scores in cohesion_scores.txt.

Submission Instructions

    Submit the following:
        Python notebook (.ipynb) with each task implemented in separate sections.
        Screenshots of the output files (message_vectors.txt, top_10_representative_vectors.txt, and cohesion_scores.txt).
        Ensure your code is well-commented for readability.
