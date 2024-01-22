from rdkit import Chem
from rdkit.Chem import BRICS
from rdkit import DataStructs
from rdkit.Chem import AllChem
import re
import numpy as np

class FP:
    def __init__(self, fp, names):
        self.fp = fp
        self.names = names

    def __str__(self):
        return "%d bit FP" % len(self.fp)

    def __len__(self):
        return len(self.fp)

def get_cfps(mol, radius=2, nBits=512, useFeatures=False, counts=False, dtype=np.float32):

    arr = np.zeros((1,), dtype)

    if counts is True:
        info = {}
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits, useFeatures=useFeatures,
                                                   bitInfo=info)
        DataStructs.ConvertToNumpyArray(fp, arr)
        arr = np.array([len(info[x]) if x in info else 0 for x in range(nBits)], dtype)
    else:
        DataStructs.ConvertToNumpyArray(
            AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits, useFeatures=useFeatures), arr)
    return FP(arr, range(nBits))

def get_Morgan(smiles):
    m = Chem.MolFromSmiles(smiles)
    Finger = get_cfps(m)
    fp = Finger.fp
    fp = fp.tolist()
    return fp
def BRICS_GetMolFrags(smi):
    mol = Chem.MolFromSmiles(smi)
    smarts = Chem.MolToSmiles(mol)
    mol = Chem.MolFromSmiles(smarts)
    #---mol Decompose
    sub_smi = BRICS.BRICSDecompose(mol)
    sub_smi = [re.sub(r'\[\d+\*\]','*',item) for item in sub_smi]
    return sub_smi, smarts
