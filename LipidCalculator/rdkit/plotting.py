import io
from typing import Iterable

from PIL import Image
from matplotlib import pyplot as plt
from rdkit import Chem
from rdkit.Chem import Mol
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

from LipidCalculator import CompoundDict
from LipidCalculator.rdkit.util import add_indices, get_combined
import logging

logger = logging.getLogger(__name__)


def mol_to_img(mol: Mol, res_pixels: int = 1000, aspect_ratio: float = 9 / 16, remove_hs: bool = True):
    w, h = res_pixels, round(aspect_ratio * res_pixels)

    drawer = rdMolDraw2D.MolDraw2DCairo(w, h)
    opts = drawer.drawOptions()
    opts.clearBackground = False  # do not paint a background (keeps PNG alpha transparent)

    if remove_hs:
        # TODO: check formula does not change
        mol_plot = Chem.RemoveHs(mol, implicitOnly=True)
        Chem.SanitizeMol(mol_plot)
        f_orig = CalcMolFormula(mol)
        f_plot = CalcMolFormula(mol_plot)
        if CompoundDict(f_orig) != CompoundDict(f_plot):
            diff = (CompoundDict(f_orig) - CompoundDict(f_plot)).formula
            logger.warning(f'while making H atoms implicit for mol with formula {f_orig}, formula changed by {diff}')
    else:
        mol_plot = mol

    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol_plot)
    drawer.FinishDrawing()
    png_bytes = drawer.GetDrawingText()

    # convert to PIL Image:
    img = Image.open(io.BytesIO(png_bytes))
    return img


def mplt_mol(mol: Mol, res_pixels=1000, ax: plt.Axes = None, **kwargs) -> plt.Axes:
    img = mol_to_img(mol, **kwargs)

    # img = Draw.MolToImage(mol, size=(res_pixels, round(9 / 16 * res_pixels)))

    if ax is None:
        fig, ax = plt.subplots(figsize=(img.width / res_pixels, img.height / res_pixels), dpi=res_pixels)
    ax.imshow(img)
    ax.axis('off')
    return ax


def plt_indices_bond(mols: Iterable[Mol] | Mol, **kwargs) -> plt.Axes:
    if isinstance(mols, Mol):
        combo = mols
    else:
        combo: Mol = get_combined(mols)
    combo = add_indices(combo)

    return mplt_mol(combo, **kwargs)
