import itertools
from collections import defaultdict
from operator import neg
import random
import math
import torch
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
import pandas as pd
import numpy as np
import os
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import DataStructs

def smiles_embedings(smiles):
    mol = Chem.MolFromSmiles(smiles)
    m = AllChem.GetMorganFingerprintAsBitVect(mol, radius=4, nBits=1024)
    # m =m.ToBitString()
    m = np.array(m).reshape(32,32)
    return m


if __name__ == "__main__":

    DATASET = "deepddi"
    with open("./data/data_DeepDDI.csv","r") as f:
        data_list = f.read().strip().split('\n')
    """Exclude data contains '.' in the SMILES format."""
    # data_list = [d for d in data_list if '.' not in d.strip().split()[0]]
    N = len(data_list)
    print(N)
    drugs1,drugs2,interactions,dbs1,dbs2= [], [], [],[],[]
    for no, data in enumerate(data_list):
        print('/'.join(map(str, [no + 1, N])))
        # db1,drug1,db2, drug2, interaction = data.strip().split(",")
        drug1,drug2, interaction = data.strip().split(",")
        d1 = smiles_embedings(drug1)
        drugs1.append(d1)
        d2 = smiles_embedings(drug2)
        drugs2.append(d2)
        interactions.append(np.array([float(interaction)]))
        # dbs1.append(db1)
        # dbs2.append(db2)
    # print(dbs1)

    dir_input = ('dataset/' + DATASET + '/ddi/')
    os.makedirs(dir_input, exist_ok=True)
    np.save(dir_input + 'drug1', drugs1)
    np.save(dir_input + 'drug2', drugs2)
    np.save(dir_input + 'interactions', interactions)
    # np.save(dir_input + 'dbs1', dbs1)
    # np.save(dir_input + 'dbs2', dbs2)
    print('The preprocess of ' + DATASET + ' dataset has finished!')
    
    
   
    
    
