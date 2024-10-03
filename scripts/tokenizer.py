import pandas as pd
from transformers import AutoTokenizer,  PreTrainedTokenizer

class Tokenizer:
    """
    A class that will contain functions and attributes tha are concerned with tokenization and alignment
    """
    def __init__(self, model_name):
        self.model_name = model_name
    
    def load_tokenizer(self, tokenizer: PreTrainedTokenizer = None):
        """
        A function that sets the class tokenizer
    
        Args:
            tokenizer(PreTrainedTokenizer): this is optional, an already initialized tokenizer
        """
        if tokenizer: self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=self.model_name)

    def tokenize_and_align(self, input: pd.DataFrame):
        """
        A function that will use the tokenizer and then align the labels

        Args:
            input(pd.DataFrame): the inputs to be tokenized and aligned
        
        Returns:
            The tokenized words and labels
        """
        tokenized_inputs = self.tokenizer(
            input['tokens'],
            truncation=True,
            is_split_into_words=True,
            padding="max_length",
            max_length=128
        )

        labels = []
        for i, label in enumerate(input['lables']):
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Get word ids for each token
            previous_word_idx = None
            label_ids = []

            for word_idx in word_ids:
                if word_idx is None:
                    # Token corresponds to special tokens like [CLS], [SEP], etc.
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    # The first token of a word
                    label_ids.append(label[word_idx])
                else:
                    # Subword token, assign -100 so it's ignored during training
                    label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)
    
        tokenized_inputs['lables'] = labels
        return tokenized_inputs