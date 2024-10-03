import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

def read_conll(path: str):
    """
    A function that will read a CoNLL formated file and return a dataframe from it.

    Args:
        path(str): the file path of the CoNLL file
    
    Returns:
        pandas dataframe of the CoNLL file
    """

    with open(path, 'r' , encoding='utf-8') as file:
        content = file.read()
    
    # get all the individual datapoints
    data_points = content.split('\n\n')

    # get all the words and labels from each data point
    final_words = []
    final_labels = []
    for data_point in data_points:
        word_label_parrings = data_point.split('\n')
        words = []
        lables = []
        for parring in word_label_parrings:
            splitting = parring.split(' ')
            word = splitting[0]
            lable = splitting[1]
            words.append(word)
            lables.append(lable)
        
        final_words.append(words)
        final_labels.append(lables)
    
    # create a dataframe
    result = pd.DataFrame()

    # add the results
    result['tokens'] = final_words
    result['lables'] = final_labels

    return result      

def compute_metrics(p):
    """
    A custom metrics function that calculates matrix
    """
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_labels = [[label for label in label_row if label != -100] for label_row in labels]
    predicted_labels = [[pred for pred, true in zip(pred_row, true_row) if true != -100] for pred_row, true_row in zip(predictions, labels)]

    # Flatten the lists
    true_labels_flat = [item for sublist in true_labels for item in sublist]
    predicted_labels_flat = [item for sublist in predicted_labels for item in sublist]

    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels_flat, predicted_labels_flat, average='weighted', zero_division=True)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }