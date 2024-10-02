import pandas as pd

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
