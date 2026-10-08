from rdkit import Chem
from rdkit.Chem import AllChem
mol = Chem.AddHs(Chem.MolFromSmiles("OCCOC(=O)c1ccc(cc1)C(=O)[O-]"))
AllChem.EmbedMolecule(mol, randomSeed=42)
AllChem.MMFFOptimizeMolecule(mol)
Chem.MolToMolFile(mol, "MHET.sdf")