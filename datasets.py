import torch
from torch.utils.data import Dataset
import re 


class Vocabulary:
  
    def __init__(self, freq_threshold=0, max_size=None):
        '''
        freq_threshold : the minimum times a word must occur in corpus to be treated in vocab
        max_size : max source vocab size. Eg. if set to 10,000, we pick the top 10,000 most frequent words and discard others
        '''
    
        self.freq_threshold = freq_threshold
        self.max_size = max_size
        
        self.itos = {0: '<PAD>', 1:'<SOS>', 2:'<EOS>', 3: '<UNK>'}  # index to string
        self.stoi = {k:j for j,k in self.itos.items()}  # string to index
    

    def __len__(self):
        return len(self.itos)
    
    
    @staticmethod
    def tokenizer(text):
        '''
        Converts the sequence to a list of tokens
        '''
        text_split = re.split('(\W+)', str(text))
        text_split = [w.strip().lower() for w in text_split if w.strip()]
        return text_split
    
    
    def build_vocabulary(self, sequence_list):
        
        '''
        Creates a mapping of index to string (self.itos) and string to index (self.stoi)
        '''

        token_frequencies = {}
        idx = len(self.itos)  # We already used 4 indexes for pad, start, end, unk
        
        # calculate freq of words
        for s in sequence_list:
            for token in self.tokenizer(s):
                if token not in token_frequencies.keys():
                    token_frequencies[token] = 1
                else:
                    token_frequencies[token] += 1


        # remove low frequency words
        token_frequencies = {k:v for k,v in token_frequencies.items() if v > self.freq_threshold} 

        # limit vocab to the max_size specified
        if self.max_size is not None:
            token_frequencies = dict(sorted(token_frequencies.items(), key = lambda x: -x[1])[:self.max_size-idx])
            
        # create vocab
        for word in token_frequencies.keys():
            self.stoi[word] = idx
            self.itos[idx] = word
            idx += 1
            
     
    def numericalize(self, text):
        '''
        Converts the list of words to a list of corresponding indexes based on self.stoi
        '''   
        # tokenize text
        tokenized_text = self.tokenizer(text)
        numericalized_text = []
        for token in tokenized_text:
            if token in self.stoi.keys():
                numericalized_text.append(self.stoi[token])
            else: #  out-of-vocab words are represented by UNK token index
                numericalized_text.append(self.stoi['<UNK>'])
                
        return numericalized_text



class TextClassifierTrainDataset(Dataset):

    def __init__(
        self, 
        df, source_column, target_column, class_names, 
        protected_attribute_column = None, protected_attribute_names = None, 
        freq_threshold = 0, max_size = None,
        vocab = None
        ):
    
        self.df = df

        # get source and target texts
        self.texts = self.df[source_column].tolist()
        self.classes = self.df[target_column].tolist()

        # build classes
        self.idx_to_class = {i:j for i, j in enumerate(class_names)}
        self.class_to_idx = {value:key for key,value in self.idx_to_class.items()}
        self.class_names = class_names

        # get protected attributes
        self.return_protected_attribute = (protected_attribute_column is not None)
        self.protected_attribute_column = protected_attribute_column
        if protected_attribute_column:
            self.protected_attributes = self.df[protected_attribute_column].tolist()
            self.idx_to_prot_attr = {i:j for i, j in enumerate(protected_attribute_names)}
            self.prot_attr_to_idx = {value:key for key,value in self.idx_to_prot_attr.items()}
        
        # build vocabulary on the input texts
        if vocab is not None:
            self.vocab = vocab
        else:
            vocab = Vocabulary(
                freq_threshold = freq_threshold,
                max_size = max_size
            )
            vocab.build_vocabulary(self.texts)
            self.vocab = vocab
        
        
    def __len__(self):
        return len(self.df)
    
    
    def __getitem__(self, index):
        '''
        __getitem__ runs on 1 example at a time. Here, we get an example at index and return its numericalized source and
        target values using the vocabulary objects we created in __init__
        '''

        text = self.texts[index]
        cls = self.classes[index]

            
        # numericalize texts: ['<SOS>', 'quick', 'brown', 'fox', <EOS>'] -> [1,12,2,9,24,2]
        numerialized_source = [self.vocab.stoi["<SOS>"]]
        numerialized_source += self.vocab.numericalize(text)
        numerialized_source.append(self.vocab.stoi["<EOS>"])

        # return text and class
        text_tensor = torch.tensor(numerialized_source)
        cls_tensor = torch.tensor(self.class_to_idx[cls]) 

        if self.return_protected_attribute:
            protected_attrib = self.protected_attributes[index]
            protected_attrib_tensor = torch.tensor(self.prot_attr_to_idx[protected_attrib]) 
            return text_tensor, cls_tensor, protected_attrib_tensor
        else:
            return text_tensor, cls_tensor


    def get_readable_tokens(self, text_tensor, cls_tensor, prot_attr_tensor=None):

        text = text_tensor.tolist()
        cls = cls_tensor.tolist()

        if prot_attr_tensor:
            protected_attrib = prot_attr_tensor.tolist()
            return [self.vocab.itos[token_id] for token_id in text], self.idx_to_class[cls], self.idx_to_prot_attr[protected_attrib]
        else:
            return [self.vocab.itos[token_id] for token_id in text], self.idx_to_class[cls]


class TextClassifierInferenceDataset(Dataset):

    def __init__(self, train_dataset, df, source_column, target_column, protected_attribute_column=None, protected_attribute_names=None, protected_attribute_source='train'):
    
        self.df = df

        # get source and target texts
        self.texts = self.df[source_column].tolist()
        self.classes = self.df[target_column].tolist()

        # get protected attribute
        if protected_attribute_source == 'self':
            self.return_protected_attribute = (protected_attribute_column is not None)
            self.protected_attribute_column = protected_attribute_column
            if protected_attribute_column:
                self.protected_attributes = self.df[protected_attribute_column].tolist()
                self.idx_to_prot_attr = {i:j for i, j in enumerate(protected_attribute_names)}
                self.prot_attr_to_idx = {value:key for key,value in self.idx_to_prot_attr.items()}
        elif protected_attribute_source == 'train':
            self.return_protected_attribute = train_dataset.return_protected_attribute
            self.protected_attribute_column = train_dataset.protected_attribute_column
            if self.return_protected_attribute:
                self.protected_attributes = self.df[self.protected_attribute_column].tolist()
                self.idx_to_prot_attr = train_dataset.idx_to_prot_attr
                self.prot_attr_to_idx = train_dataset.prot_attr_to_idx
        
        # save vocab
        self.vocab = train_dataset.vocab
        self.class_to_idx = train_dataset.class_to_idx
        self.idx_to_class = train_dataset.idx_to_class

        
        
    def __len__(self):
        return len(self.df)
    
    
    def __getitem__(self, index):
        '''
        __getitem__ runs on 1 example at a time. Here, we get an example at index and return its numericalized source and
        target values using the vocabulary objects we created in __init__
        '''

        text = self.texts[index]
        cls = self.classes[index]
            
        # numericalize texts: ['<SOS>', 'quick', 'brown', 'fox', <EOS>'] -> [1,12,2,9,24,2]
        numerialized_source = [self.vocab.stoi["<SOS>"]]
        numerialized_source += self.vocab.numericalize(text)
        numerialized_source.append(self.vocab.stoi["<EOS>"])

        # return text and class
        text_tensor = torch.tensor(numerialized_source)
        cls_tensor = torch.tensor(self.class_to_idx[cls]) 

        if self.return_protected_attribute:
            protected_attrib = self.protected_attributes[index]
            protected_attrib_tensor = torch.tensor(self.prot_attr_to_idx[protected_attrib]) 
            return text_tensor, cls_tensor, protected_attrib_tensor
        else:
            return text_tensor, cls_tensor


    def get_readable_tokens(self, text_tensor, cls_tensor, prot_attr_tensor=None):

        text = text_tensor.tolist()
        cls = cls_tensor.tolist()

        if prot_attr_tensor:
            protected_attrib = prot_attr_tensor.tolist()
            return [self.vocab.itos[token_id] for token_id in text], self.idx_to_class[cls], self.idx_to_prot_attr[protected_attrib]
        else:
            return [self.vocab.itos[token_id] for token_id in text], self.idx_to_class[cls]