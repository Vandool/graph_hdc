import gzip
import math
import pickle

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from src.utils.utils import GLOBAL_DATASET_PATH


# Load the fragment scores
def readFragmentScores(name=GLOBAL_DATASET_PATH / "sa_score/fpscores.pkl.gz"):
    with gzip.open(name, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    outDict = {}
    for i in data:
        outDict[i[0]] = float(i[1])
    return outDict


fragmentScore = None


def numBridgeheadsAndSpiro(mol, ri=None):
    nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    nBridge = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    return nBridge, nSpiro


def calculateScore(m):
    global fragmentScore
    if fragmentScore is None:
        fragmentScore = readFragmentScores()
    # fragment score
    fp = rdMolDescriptors.GetMorganFingerprint(m, 2)
    fps = fp.GetNonzeroElements()
    score1 = 0.0
    nf = 0
    for bitId, v in fps.items():
        nf += v
        sfp = fragmentScore.get(bitId, -4)
        score1 += sfp * v
    score1 /= nf

    # features score
    nAtoms = m.GetNumAtoms()
    nChiralCenters = len(Chem.FindMolChiralCenters(m, includeUnassigned=True))
    ri = m.GetRingInfo()
    nBridgeheads, nSpiro = numBridgeheadsAndSpiro(m, ri)
    nMacrocycles = 0
    for x in ri.AtomRings():
        if len(x) > 8:
            nMacrocycles += 1

    sizePenalty = nAtoms**1.005 - nAtoms
    stereoPenalty = math.log10(nChiralCenters + 1)
    spiroPenalty = math.log10(nSpiro + 1)
    bridgePenalty = math.log10(nBridgeheads + 1)
    macrocyclePenalty = 0.0
    if nMacrocycles > 0:
        macrocyclePenalty = math.log10(2)

    score2 = 0.0 - sizePenalty - stereoPenalty - spiroPenalty - bridgePenalty - macrocyclePenalty

    # correction for fingerprint density
    # not in the original publication, added in this version
    score3 = 0.0
    if nAtoms > len(fps):
        score3 = math.log(float(nAtoms) / len(fps)) * 0.5

    sascore = score1 + score2 + score3

    # normalization
    min = -4.0
    max = 2.5
    sascore = 11.0 - ((sascore - min + 1) / (max - min) * 9.0)
    # smooth the 10-end
    if sascore > 8.0:
        sascore = 8.0 + math.log(sascore + 1.0 - 9.0)
    if sascore > 10.0:
        sascore = 10.0
    elif sascore < 1.0:
        sascore = 1.0

    return sascore
