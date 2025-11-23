import io

from PIL import Image
from matplotlib import pyplot as plt
from rdkit import Chem
from rdkit.Chem import Mol
from rdkit.Chem.Draw import rdMolDraw2D

from LipidCalculator.rdkit.util import add_indices, get_combined


def mol_to_img(mol: Mol, res_pixels: int = 1000, aspect_ratio: float = 9 / 16):
    w, h = res_pixels, round(aspect_ratio * res_pixels)

    drawer = rdMolDraw2D.MolDraw2DCairo(w, h)
    opts = drawer.drawOptions()
    opts.clearBackground = False  # do not paint a background (keeps PNG alpha transparent)

    rdMolDraw2D.PrepareAndDrawMolecule(drawer, Chem.RemoveHs(mol, implicitOnly=True))
    drawer.FinishDrawing()
    png_bytes = drawer.GetDrawingText()

    # convert to PIL Image:
    img = Image.open(io.BytesIO(png_bytes))
    return img


def mplt_mol(mol: Mol, res_pixels=1000, **kwargs):
    img = mol_to_img(mol, **kwargs)

    # img = Draw.MolToImage(mol, size=(res_pixels, round(9 / 16 * res_pixels)))

    fig, ax = plt.subplots(figsize=(img.width / res_pixels, img.height / res_pixels), dpi=res_pixels)
    ax.imshow(img)
    ax.axis('off')
    return fig, ax


def plt_indices_bond(mols):
    combo: Mol = get_combined(mols)
    combo = add_indices(combo)

    mplt_mol(combo)
    plt.show()
