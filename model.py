import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch import Tensor
from torch.nn import Transformer
import math


class PositionalEncoding(nn.Module):
    def __init__(
        self, 
        emb_size: int, 
        dropout: float, 
        maxlen: int = 5000
    ):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


class TokenEmbedding(nn.Module):
    def __init__(
        self, 
        vocab_size: int, 
        emb_size
    ):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class TransformerClassifier(nn.Module):

    def __init__(
        self,
        num_encoder_layers: int,
        emb_size: int,
        nhead: int,
        vocab_size: int,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        return_feature_vector: bool = False
    ):
        super(TransformerClassifier, self).__init__()
        
        self.src_tok_emb = TokenEmbedding(vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout = dropout)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=emb_size, nhead=nhead), 
            num_layers = num_encoder_layers
        )
        # self.flatten = nn.Flatten(start_dim=1, end_dim=2)
        self.fc1 = nn.Linear(emb_size, dim_feedforward)
        self.fc1_activation = nn.Tanh()
        self.fc1_dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(dim_feedforward, 2)

        # return intermediate feature vector (layer before logits layer)?
        self.return_feature_vector = return_feature_vector

    def forward(
        self,
        src: Tensor
    ):
        '''
        Mean aggregation on encoder output for classification purposes
        '''
        src_emb = self.src_tok_emb(src)
        src_emb_enc = self.positional_encoding(src_emb)
        encoder_out = self.transformer_encoder(src_emb_enc)

        encoder_agg = encoder_out.mean(0)  # take mean over sequence length
        feature_vec = self.fc1(encoder_agg)
        feature_vec_activation = self.fc1_activation(feature_vec)
        feature_vec_dropout = self.fc1_dropout(feature_vec_activation)
        logits = self.classifier(feature_vec_dropout) 

        if self.return_feature_vector:
            return logits, feature_vec
        else:
            return logits


class DemographicDiscriminator(nn.Module):
    def __init__(
        self, 
        input_dim : int, 
        hidden_size: int, 
        num_classes: int,
        dropout
    ):
        super(DemographicDiscriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim , hidden_size)
        self.fc1_activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h1 = self.fc1(x)
        h1_activation = self.fc1_activation(h1)
        h1_dropout = self.dropout(h1_activation)
        logits = self.classifier(h1_dropout)
        return logits


class CombinedModel(nn.Module):
    def __init__(self, mainClassifier, adversarialDiscriminator):
        super(CombinedModel, self).__init__()
        self.mainClassifier = mainClassifier
        self.adversarialDiscriminator = adversarialDiscriminator
        
    def forward(self, x):
        mainClassifierLogits, mainClassifierFeatureVec = self.mainClassifier(x)
        adversarialDiscriminatorLogits = self.adversarialDiscriminator(mainClassifierFeatureVec)
        return mainClassifierLogits, adversarialDiscriminatorLogits