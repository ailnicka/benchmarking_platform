#
# $Id$
#
# module to calculate a fingerprint from SMILES

from os import fdopen
from numpy.lib.index_tricks import fill_diagonal
from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem
from rdkit.Avalon import pyAvalonTools as fpAvalon
from rdkit.Chem.AtomPairs import Pairs, Torsions
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem.ChemicalFeatures import BuildFeatureFactory
from rdkit.Chem import rdMolDescriptors
from tensorflow.keras.models import load_model
import numpy as np

# implemented fingerprints:
# ECFC0 (ecfc0), ECFP0 (ecfp0), MACCS (maccs), 
# atom pairs (ap), atom pairs bit vector (apbv), topological torsions (tt)
# hashed atom pairs (hashap), hashed topological torsions (hashtt) --> with 1024 bits
# ECFP4 (ecfp4), ECFP6 (ecfp6), ECFC4 (ecfc4), ECFC6 (ecfc6) --> with 1024 bits
# FCFP4 (fcfp4), FCFP6 (fcfp6), FCFC4 (fcfc4), FCFC6 (fcfc6) --> with 1024 bits
# Avalon (avalon) --> with 1024 bits
# long Avalon (laval) --> with 16384 bits
# long ECFP4 (lecfp4), long ECFP6 (lecfp6), long FCFP4 (lfcfp4), long FCFP6 (lfcfp6) --> with 16384 bits
# RDKit with path length = 5 (rdk5), with path length = 6 (rdk6), with path length = 7 (rdk7)
# 2D pharmacophore (pharm) ?????????????

nbits = 2048
longbits = 16384

def compress_via_int(fp, int_len = 8, to_float=False):
    if int_len not in (8, 16, 32):
        raise ValueError('Int_len argument shall be 8, 16 or 32')
    
    len_comp_fp = int(np.ceil(len(fp)/int_len))    
    def bits_to_ints(fp):
        # Makes signed int
        max_int = 2**(int_len-1)
        bitstr = ''.join((str(i) for i in fp))
        return np.array([max_int - int(bitstr[i*int_len : (i+1)*int_len], 2) for i in range(len_comp_fp)])
    
    compressed_fp = bits_to_ints(fp)
    if to_float:
        "Note: sigmoid is pretty bad though, cause it works best on -5,5, but further these would be mostly 0s and 1s"
        compressed_fp = 1.0/(1.0 + np.exp(-compressed_fp))
    return compressed_fp

def compress_via_logic(fp, window, op='or'):
    OP_DICT = {'and': all, 'or': any, 'sum': sum}

    if op not in OP_DICT.keys():
        raise ValueError('Op value shall be "and", "or" or "sum"')
    
    len_comp_fp = int(np.ceil(len(fp)/window))
    def operation_on_bits(fp):
        return [int(OP_DICT[op](fp[i*window: (i+1)*window])) for i in range(len_comp_fp)]
    compressed_fp = np.array(operation_on_bits(fp))

    return compressed_fp


class AEFingerprints:
    def __init__(self):
        #core = "/Users/ailnicka/PycharmProjects/"
        core = "/cluster/work/schneider/modlab/ailnicka/"
        self.MACCS_50_np_compressor = load_model(core+"AE/Models/MACCS_50_no_prop_01")
        self.Morgan_2_np_compressor = load_model(core+"AE/Models/Morgan_2_100_no_prop_01")
        self.MACCS_50_compressor = load_model(core+"AE/Models/MACCS_50_01")
        self.Morgan_2_compressor = load_model(core+"AE/Models/Morgan_2_100_01")
        self.MACCS_Morgan_compressor = load_model(core+"AE/Models/double_MACCS_Morgan2_100_01")
        self.MACCS_Morgan_np_compressor = load_model(core+"AE/Models/double_MACCS_Morgan2_no_prop_01")

    def compressed_MACCS_50(self, m):
        m = MACCSkeys.GenMACCSKeys(m)
        return self.MACCS_50_compressor.encoder(np.array([m])).numpy().flatten()

    def compressed_Morgan2_100(self, m):
        m = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=nbits)
        return self.Morgan_2_compressor.encoder(np.array([m])).numpy().flatten()
    
    def compressed_MACCS_50_np(self, m):
        m = MACCSkeys.GenMACCSKeys(m)
        return self.MACCS_50_np_compressor.encoder(np.array([m])).numpy().flatten()

    def compressed_Morgan2_100_np(self, m):
        m = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=nbits)
        return self.Morgan_2_np_compressor.encoder(np.array([m])).numpy().flatten()

    def compressed_MACCS_Morgan2_100(self, m):
        m1 = MACCSkeys.GenMACCSKeys(m)
        m2 = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=nbits)
        return self.MACCS_Morgan_compressor.encoder([np.array([m1]), np.array([m2])]).numpy().flatten()

    def compressed_MACCS_Morgan2_100_np(self, m):
        m1 = MACCSkeys.GenMACCSKeys(m)
        m2 = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=nbits)
        return self.MACCS_Morgan_np_compressor.encoder([np.array([m1]), np.array([m2])]).numpy().flatten()

aef = AEFingerprints()

# dictionary
fpdict = {}
fpdict['ecfp0'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 0, nBits=nbits)
fpdict['ecfp2'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 1, nBits=nbits)
fpdict['ecfp4'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=nbits)
fpdict['ecfp6'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits=nbits)
fpdict['ecfc0'] = lambda m: AllChem.GetMorganFingerprint(m, 0)
fpdict['ecfc2'] = lambda m: AllChem.GetMorganFingerprint(m, 1)
fpdict['ecfc4'] = lambda m: AllChem.GetMorganFingerprint(m, 2)
fpdict['ecfc6'] = lambda m: AllChem.GetMorganFingerprint(m, 3)
fpdict['fcfp2'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 1, useFeatures=True, nBits=nbits)
fpdict['fcfp4'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, useFeatures=True, nBits=nbits)
fpdict['fcfp6'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, useFeatures=True, nBits=nbits)
fpdict['fcfc2'] = lambda m: AllChem.GetMorganFingerprint(m, 1, useFeatures=True)
fpdict['fcfc4'] = lambda m: AllChem.GetMorganFingerprint(m, 2, useFeatures=True)
fpdict['fcfc6'] = lambda m: AllChem.GetMorganFingerprint(m, 3, useFeatures=True)
fpdict['lecfp4'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=longbits)
fpdict['lecfp6'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits=longbits)
fpdict['lfcfp4'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2, useFeatures=True, nBits=longbits)
fpdict['lfcfp6'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 3, useFeatures=True, nBits=longbits)
fpdict['maccs'] = lambda m: MACCSkeys.GenMACCSKeys(m)
fpdict['ap'] = lambda m: Pairs.GetAtomPairFingerprint(m)
fpdict['tt'] = lambda m: Torsions.GetTopologicalTorsionFingerprintAsIntVect(m)
fpdict['hashap'] = lambda m: rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(m, nBits=nbits)
fpdict['hashtt'] = lambda m: rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(m, nBits=nbits)
fpdict['avalon'] = lambda m: fpAvalon.GetAvalonFP(m, nbits)
fpdict['laval'] = lambda m: fpAvalon.GetAvalonFP(m, longbits)
fpdict['rdk5'] = lambda m: Chem.RDKFingerprint(m, maxPath=5, fpSize=nbits, nBitsPerHash=2)
fpdict['rdk6'] = lambda m: Chem.RDKFingerprint(m, maxPath=6, fpSize=nbits, nBitsPerHash=2)
fpdict['rdk7'] = lambda m: Chem.RDKFingerprint(m, maxPath=7, fpSize=nbits, nBitsPerHash=2)
fpdict['comp_MACCS'] = lambda m: aef.compressed_MACCS_50(m)
fpdict['comp_Morgan2'] = lambda m: aef.compressed_Morgan2_100(m)
fpdict['comp_MACCS_np'] = lambda m: aef.compressed_MACCS_50_np(m)
fpdict['comp_Morgan2_np'] = lambda m: aef.compressed_Morgan2_100_np(m)
fpdict['comp_MACCS_Morgan2'] = lambda m: aef.compressed_MACCS_Morgan2_100(m)
fpdict['comp_MACCS_Morgan2_np'] = lambda m: aef.compressed_MACCS_Morgan2_100_np(m)
fpdict['Morgan2'] = lambda m: AllChem.GetMorganFingerprintAsBitVect(m, 2)
fpdict['bench_comp_int_MACCS'] = lambda m: compress_via_int(list(MACCSkeys.GenMACCSKeys(m)))
fpdict['bench_comp_and_MACCS'] = lambda m: compress_via_logic(list(MACCSkeys.GenMACCSKeys(m)), 10, 'and')
fpdict['bench_comp_or_MACCS'] = lambda m: compress_via_logic(list(MACCSkeys.GenMACCSKeys(m)), 10, 'or')
fpdict['bench_comp_sum_MACCS'] = lambda m: compress_via_logic(list(MACCSkeys.GenMACCSKeys(m)), 10, 'sum')
fpdict['bench_comp_int_Morgan2'] = lambda m: compress_via_int(list(AllChem.GetMorganFingerprintAsBitVect(m, 2)))
fpdict['bench_comp_and_Morgan2'] = lambda m: compress_via_logic(list(AllChem.GetMorganFingerprintAsBitVect(m, 2)), 10, 'and')
fpdict['bench_comp_or_Morgan2'] = lambda m: compress_via_logic(list(AllChem.GetMorganFingerprintAsBitVect(m, 2)), 10, 'or')
fpdict['bench_comp_sum_Morgan2'] = lambda m: compress_via_logic(list(AllChem.GetMorganFingerprintAsBitVect(m, 2)), 10, 'sum')


def CalculateFP(fp_name, smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        raise ValueError('SMILES cannot be converted to a RDKit molecules:', smiles)

    return fpdict[fp_name](m)
