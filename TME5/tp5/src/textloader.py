import sys
import unicodedata
import string
from typing import List
from torch.utils.data import Dataset, DataLoader
import torch
import re

## Token de padding (BLANK) represente l'index du padding token
PAD_IX = 0
## Token de fin de séquence
EOS_IX = 1

# 'LETTRES' is a string containing ASCII letter, punctuation, digits and space
LETTRES = string.ascii_letters + string.punctuation + string.digits + ' '

# MAPPING FROM INDEX TO CHARACTER 
# Dictionnary where keys are indices starting from 2 and increasing up to the length of 'LETTRES' +1
## and values are characters from 'LETTERS'
id2lettre = dict(zip(range(2, len(LETTRES)+2), LETTRES))

# This mapping is used to convert an index to its corresponding character
id2lettre[PAD_IX] = '<PAD>' ##NULL CHARACTER
id2lettre[EOS_IX] = '<EOS>'

# MAPPING FROM CHARACTER TO INDEX
# Dictionnary where keys are characters from 'LETTRES' and 
## values are the corresponding indices from 'id2lettre'
lettre2id = dict(zip(id2lettre.values(),id2lettre.keys()))

def normalize(s):
    """ enlève les accents et les caractères spéciaux"""
    return ''.join(c for c in unicodedata.normalize('NFD', s) if  c in LETTRES)

def string2code(s):
    """prend une séquence de lettres et renvoie la séquence d'entiers correspondantes"""
    return torch.tensor([lettre2id[c] for c in normalize(s)])

def code2string(t):
    """ prend une séquence d'entiers et renvoie la séquence de lettres correspondantes """
    if type(t) !=list:
        t = t.tolist()
    return ''.join(id2lettre[i] for i in t)

class TextDataset(Dataset):
    def __init__(self, text: str, *, maxsent=None, maxlen=None):
        """  Dataset pour les tweets de Trump
            * fname : nom du fichier
            * maxsent : nombre maximum de phrases.
            * maxlen : longueur maximale des phrases.
        """
        maxlen = maxlen or sys.maxsize
        self.phrases = [re.sub(' +',' ',p[:maxlen]).strip() +"." for p in text.split(".") if len(re.sub(' +',' ',p[:maxlen]).strip())>0]
        if maxsent is not None:
            self.phrases=self.phrases[:maxsent]
        self.maxlen = max([len(p) for p in self.phrases])

    def __len__(self):
        return len(self.phrases)

    def __getitem__(self, i):
        return string2code(self.phrases[i])

def pad_collate_fn(samples: List[List[int]]):
    #  TODO:  Renvoie un batch à partir d'une liste de listes d'indexes (de phrases) qu'il faut padder.
    '''
    Prépare un tenseur batch à partir d'une liste d'exemple du TextDataset
    Doit ajouter le code du symbole EOS à la fin de chaque exemple et
    Padder les séquences avec le code du caractère nul 
    '''
    # Ajout du token EOS à la fin de chaque séquence
    samples = [torch.cat((torch.tensor(seq), torch.tensor([EOS_IX]))) for seq in samples]
    
    # Calculer la longueur maximale du batch
    max_len = max(len(seq) for seq in samples)

    #Initialiser le batch tensor avec le token du caractère nul
    batch = [seq.tolist() + [PAD_IX] * (max_len - len(seq)) for seq in samples]

    return torch.Tensor(batch).T



if __name__ == "__main__":
    test = "C'est. Un. Test."
    ds = TextDataset(test)
    loader = DataLoader(ds, collate_fn=pad_collate_fn, batch_size=3)
    data = next(iter(loader))
    print("Chaîne à code : ", test)
    # Longueur maximum
    assert data.shape == (7, 3)
    print("Shape ok")
    # e dans les deux cas
    assert data[2, 0] == data[1, 2]
    print("encodage OK")
    # Token EOS présent
    assert data[5,2] == EOS_IX
    print("Token EOS ok")
    # BLANK présent
    assert (data[4:,1]==0).sum() == data.shape[0]-4
    print("Token BLANK ok")
    # les chaînes sont identiques
    s_decode = " ".join([code2string(s).replace(id2lettre[PAD_IX],"").replace(id2lettre[EOS_IX],"") for s in data.t()])
    print("Chaîne décodée : ", s_decode)
    assert test == s_decode
    # " ".join([code2string(s).replace(id2lettre[PAD_IX],"").replace(id2lettre[EOS_IX],"") for s in data.t()])
