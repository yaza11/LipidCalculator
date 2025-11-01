"""Use alpha/induced cleavages (breakage of carbon-carbon bonds next to
heteroatoms) to predict fragments."""
import warnings
from typing import Literal

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from rdkit import Chem
from rdkit.Chem import Mol, Draw
from rdkit.Chem.Descriptors import ExactMolWt

from LipidCalculator.compound_groups.intact_polar_lipids.generate_ipl import plt_indices_bond
from LipidCalculator.compound_groups.common_fragments import MASS_PROTON

CLEAVAGE_POS: list[str] = ['CO', 'CN', 'CP', 'CS']
CleavagePos: type = Literal[*CLEAVAGE_POS]


def find_cleavage_bonds(mol: Mol) -> dict[CleavagePos, tuple[tuple[int, int], ...]]:
    """
    For a given molecule, identify bonds that are prone to inductive cleavages.
    This applies to Carbon atoms bonded to O, N, S(?), P
    """
    matches = {}
    for s_pattern in CLEAVAGE_POS:
        pattern = Chem.MolFromSmarts(s_pattern)
        matches[s_pattern] = mol.GetSubstructMatches(pattern)

    return matches


def split_bond(mol, atom_idx1, atom_idx2):
    """Cleave a molecule at the bond between the specified atoms. Return the generated fragments."""
    rw_mol = Chem.RWMol(mol)
    atom1 = rw_mol.GetAtomWithIdx(atom_idx1)
    atom2 = rw_mol.GetAtomWithIdx(atom_idx2)
    rw_mol.RemoveBond(atom1.GetIdx(), atom2.GetIdx())
    Chem.SanitizeMol(rw_mol)
    frags = Chem.GetMolFrags(rw_mol, asMols=True, sanitizeFrags=True)
    return frags


def split_bond_add_H(mol, atom_idx1, atom_idx2, plts: bool = False) -> dict[str, Mol]:
    """Split molecules at the specified indices. Assumes that atom 1 is a C and
    atom 2 a heteroatom. Heteroatom will get H atom bonded (in reality, it is
    assumed that heteroatom was protonated before cleavage) and C atom obtains
    formal charge"""
    # Clone the molecule
    rw_mol = Chem.RWMol(mol)

    # Ensure atom1 is carbon, atom2 is heteroatom
    atom1 = rw_mol.GetAtomWithIdx(atom_idx1)
    atom2 = rw_mol.GetAtomWithIdx(atom_idx2)

    # Prepare carbon to accept explicit H+
    atom1.SetFormalCharge(1)

    # Remove the bond
    rw_mol.RemoveBond(atom1.GetIdx(), atom2.GetIdx())

    # protonate one of the H atoms bonded to C
    # heteroatom has adduct that is now no longer charged
    h_atom = Chem.Atom(1)
    h_idx2 = rw_mol.AddAtom(h_atom)
    rw_mol.AddBond(atom2.GetIdx(), h_idx2, Chem.BondType.SINGLE)

    if plts:
        plt_indices_bond([rw_mol])

    # Sanitize before fragmenting
    Chem.SanitizeMol(rw_mol)

    # Get fragments as separate Mols
    frags = Chem.GetMolFrags(rw_mol, asMols=True, sanitizeFrags=True)
    out = {}
    for frag in frags:
        c = Chem.GetFormalCharge(frag)
        if c == 0:
            out['neutral'] = frag
        elif c > 0:
            out['pos'] = frag
        else:
            out['neg'] = frag
    return out  # tuple of Mol objects


def set_formal_charge_zero(mol):
    rw_mol = Chem.RWMol(mol)  # make mutable
    for atom in rw_mol.GetAtoms():
        atom.SetFormalCharge(0)
    Chem.SanitizeMol(rw_mol)  # recheck valences, update
    return rw_mol.GetMol()


def add_proton_to_heteroatom(mol):
    # make a mutable copy of the molecule
    rw_mol = Chem.RWMol(mol)

    # Find all heteroatoms (not C or H)
    heteroatoms = [
        atom for atom in rw_mol.GetAtoms()
        if atom.GetAtomicNum() not in [6, 1]
           and atom.GetFormalCharge() <= 0  # avoid overcharged species
           and atom.GetTotalValence() <= Chem.GetPeriodicTable().GetDefaultValence(atom.GetAtomicNum())
    ]
    if len(heteroatoms) == 0:
        heteroatoms = [
            atom for atom in rw_mol.GetAtoms()
            if atom.GetAtomicNum() not in [1]
               and atom.GetFormalCharge() <= 0  # avoid overcharged species
               and atom.GetTotalValence() <= Chem.GetPeriodicTable().GetDefaultValence(atom.GetAtomicNum())
        ]

    # Pick the first heteroatom
    heteroatom = heteroatoms[0]
    het_idx = heteroatom.GetIdx()

    # Add a hydrogen atom
    hydrogen = Chem.Atom(1)
    hydrogen.SetFormalCharge(1)  # protonated hydrogen
    h_idx = rw_mol.AddAtom(hydrogen)  # returns new atom index

    # Add a single bond between heteroatom and hydrogen
    rw_mol.AddBond(het_idx, h_idx, Chem.BondType.SINGLE)

    # Optionally set the formal charge on the heteroatom to +1
    heteroatom.SetFormalCharge(heteroatom.GetFormalCharge() + 1)

    # Convert back to immutable Mol and sanitize to update valences and implicit Hs
    mol_result = rw_mol.GetMol()
    Chem.SanitizeMol(mol_result)  # recomputes valences, implicit H counts etc.

    return mol_result


def get_fragments(
        mol: Mol,
        keep_neutral=False,
        max_recursion_depth=3
) -> list[tuple[Mol, Mol]]:
    """Split mol at cleavage pos. C atom gets positive charge.

    :returns
    a list of tuples in which the first entry is the positive and the second the neutral fragment.
    """
    fragments: list[tuple[Mol, Mol]] = []

    cleavage_pos = find_cleavage_bonds(mol)
    for k, bonds in cleavage_pos.items():
        for bond in bonds:
            if keep_neutral:
                frags: tuple[Mol, Mol] = split_bond(mol, *bond)
            else:
                frags_d = split_bond_add_H(mol, *bond)
                frags = frags_d.get('pos'), frags_d.get('neutral')
            fragments.append(frags)
            if not keep_neutral and (max_recursion_depth > 0):
                # positive fragments can be fragmented further (and show up at the detector)
                frag = set_formal_charge_zero(frags[0])
                # set formal charge to 0

                fragments.extend(
                    get_fragments(frag, keep_neutral=keep_neutral, max_recursion_depth=max_recursion_depth - 1))

    return fragments


def plot_ms2_prediction(mol, fragments: dict[float, Mol], annotations: dict[float, str] = None):
    """Draw the original molecule, the generated fragments and the fragment pattern."""
    if annotations is None:
        annotations = {k: None for k in fragments}
    else:
        # TODO: check annotations match fragments
        pass
    mzs = fragments.keys()
    frag_mols = fragments.values()

    res_pixels = 1000
    img = Draw.MolToImage(Chem.RemoveHs(mol), size=(res_pixels, round(9 / 16 * res_pixels)))
    # fig, axs = plt.subplots(nrows=2, dpi=res_pixels)
    fig, axs = plt.subplots(nrows=2, layout='tight')
    axs[0].imshow(img)
    axs[0].axis('off')

    axs[1].stem(mzs, [1] * len(mzs), markerfmt='')
    for mz, frag, ann in zip(mzs, frag_mols, annotations.values()):
        # axs[1].text(mz, 1, d, rotation=45)
        inset_ax = inset_axes(axs[1], width=.8, height=.8, loc='lower center',
                              bbox_to_anchor=(mz, 1.1),
                              bbox_transform=axs[1].get_xaxis_transform(),
                              borderpad=0)
        lbl_mz = f'{mz:.4f}'
        txt = lbl_mz if ann is None else ann + '\n' + f'({lbl_mz})'
        axs[1].text(mz, .8, txt, rotation=45)
        frag_img = Draw.MolToImage(Chem.RemoveHs(frag), size=(200, 200))
        inset_ax.imshow(frag_img)
        inset_ax.axis('off')

    axs[1].set_xlabel('fragment m/z in Da')
    return fig, axs


def predict_ms2(smiles: str, adduct=None, **kwargs) -> dict[float, Mol]:
    """Generate the fragment spectrum for a smiles. Assumes molecule is protonated."""
    mol: Mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    frags = get_fragments(mol, **kwargs)
    # pos_frags: list[Mol] = [t[0] for t in frags]

    # TODO: implement for different adduct types
    if adduct is None:
        adduct_mass = MASS_PROTON

    desc = [add_proton_to_heteroatom(mol)]
    mzs = [ExactMolWt(desc[0])]
    for t in frags:
        frag = t[0]
        mz, d = ExactMolWt(frag), frag
        mzs.append(mz)
        desc.append(d)

    return dict(zip(mzs, desc))


def predict_losses(smiles: str, **kwargs) -> dict[float, Mol]:
    mol: Mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    frags = get_fragments(mol, **kwargs)

    desc = [add_proton_to_heteroatom(mol)]
    mzs = [ExactMolWt(desc[0])]
    for t in frags:
        frag = t[1]
        if frag is None:
            continue
        mz = ExactMolWt(frag)
        mzs.append(mz)
        desc.append(frag)

    return dict(zip(mzs, desc))


def get_neutral_equivalents_for_predicted_ms2(smiles):
    """Instead of the protonated fragments, this returns the neutral molecules
    for better substructure match"""
    mol: Mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    cleavage_pos = find_cleavage_bonds(mol)
    predicted = predict_ms2(smiles)
    neutral_frags_conv: list[tuple[Mol, Mol]] = get_fragments(mol, cleavage_pos, keep_neutral=True)
    # deconvolute
    neutral_frags: list[Mol] = []
    for neuts in neutral_frags_conv:
        neutral_frags.extend(neuts)

    neutral_equivalents = {}
    for mz, mol in predicted.items():
        dists = np.array([np.abs((mz - MASS_PROTON) - ExactMolWt(neut))
                          for neut in neutral_frags])
        closest_idx = np.argmin(dists)
        print(dists, closest_idx)
        dist = dists[closest_idx]
        if dist > 1e-3:
            warnings.warn(f'found large deviation for {smiles=} and {mz=}')
        neutral_equivalents[mz] = neutral_frags[closest_idx]
    return neutral_equivalents


def test_inductive_cleavage_simple():
    smiles = "CCOCC"  # Contains O, N (heteroatoms)
    mol: Mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    plt_indices_bond([mol])
    frags = split_bond_add_H(mol, 1, 2, plts=True)


if __name__ == "__main__":
    # Example
    # smiles = "CCOCC"  # Contains O, N (heteroatoms)
    # smiles = 'CCCC=CCCC=CCC=CCCC=CCCCC(=O)OCC(COC1OC(COC2OC(CO)C(O)C(O)C2O)C(O)C(O)C1O)OC(=O)CCCCCCCCCCCCCCCCC'  # 2G,DAG,C18:0,C20:4
    # smiles = 'CCCCCCCCCCCCCCCOCC(COP(=O)(O)OCCN)OCCCCCCCCCCCCCCC'  # PE,DEG,C30:0
    smiles = 'CC(C)CCCC(C)CCCC(C)CCCC(C)CCOCC(COC1OC(COC2OC(CO)C(O)C(O)C2O)C(O)C(O)C1O)OCCC(C)CCCC(C)CCCC(C)CCCC(C)C'  # 2G AR
    # mol: Mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    # plt_indices_bond([mol])
    # cleavage_pos = find_cleavage_bonds(mol)
    # # split_bond_add_H(mol, 34, 35, plts=True)
    # frags = get_fragments(mol, cleavage_pos)

    ms = predict_ms2(smiles, max_recursion_depth=2)
    mol = Chem.MolFromSmiles(smiles)

    fig, axs = plot_ms2_prediction(mol, ms)
    plt.show()
