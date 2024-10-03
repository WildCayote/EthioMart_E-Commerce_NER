import re
import pandas as pd

class NERLabeler:
    """
    A class that serves as the wrapper/container of functions that help label Amharic text to a specified NER label set.
    Currently the label sets are one of product, location and price found in a given text.
    """
    def __init__(self):
        self.product_keywords = ["Jacket", "jacket", "Sweater", "sweater", "ጫማዎች", "ቆዳ", "መዳመጫ", "ማብሰያ", "መያዣ", "ብርጭቆ", "ቦይለር"]
        self.location_keywords = ["Addis Ababa", "AA", "Mall", "mall", "Floor", "floor", "HayaHulet", "Hayahulet", "አዲስ አበባ", "ቦሌ", "መደሐንያለም", "ህንፃ", "ፎቅ"]
        self.price_keywords = ["br", "Br", "ETB", "etb", "Price", "price", "በ", "ከ"]

    def label_text(self, text: str):
        """
        A function that will label the part of text as per their NER labels.

        Args:
            text(str): the text to be labled
        Returns:
            A tuple with two list, one that contains the words and the other contains their NER lables
        """

        # loop through all the words in the passed text
        words = text.split(' ')
        lables = []

        for index, word in enumerate(words):
            word_stripped = word.strip()

            # check if the word is numeric
            if word_stripped.isnumeric():
                try:
                    next_token = words[index - 1]
                    if next_token in self.price_keywords:
                        lables.append("I-PRICE")
                        continue
                except Exception as e:
                    pass
            
            # check if the word in one of price_keywords
            if word_stripped in self.price_keywords:
                if word_stripped in ["በ", "ከ"]: lables.append("B-PRICE")
                else: lables.append("I-PRICE")
                continue
            
            # check for location
            if word_stripped in self.location_keywords:
                lables.append("B-LOCATION")
                continue

            # check for product
            if word_stripped in self.product_keywords:
                lables.append("B-Product")
                continue
            
            # if all fails just lable it as 'O'
            lables.append("O")
        
        # combine the word and labels
        zipped = zip(words, lables)
        combined = [f"{word} {lable}" for word, lable in zipped]

        return combined
    
    def save_conll(self, df: pd.DataFrame, col: str, path: str):
        """
        A function that will save the the given CoNLL found in a dataframe to a text file.

        Args:
            df(pd.DataFrame): the dataframe to be saved
            col(str): the column that contains the word an label pairings (tuples)
            path(str): the path to export the CoNLL file to
        
        Returns:
            Returns the final string that was saved
        """
        labelings = df[col].to_list()
        result = []
        for label in labelings:
            formated_text = '\n'.join(label)
            result.append(formated_text)
        # join each sentence with a newline in between them
        result = '\n\n'.join(result)
        
        with open(path , 'w', encoding='utf-8') as file:
            file.write(result)
        
        return result


if __name__ == "__main__":
    import torch

    print(torch.cuda.is_available())
    print(torch.cuda.device_count())

    print(f"Using device: {torch.cuda.get_device_name(0)}")