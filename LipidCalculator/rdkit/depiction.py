"""
Depictions of lipids with a normalized layout: chains are drawn as horizontal zigzag lines, the rest of the molecule
(polyols, rings, branches, head groups) is placed around them by a spring layout. Head groups can be collapsed into a
label and chains can be shortened.

LAYOUT
======
- chains are the connected parts of pure hydrocarbon carbons (no bond to a heteroatom), their backbone is the longest
  path through them (it passes the longer side of rings), extended by up to two atoms towards the rest of the
  molecule (e.g. the ether O and C1, or the ester O and carbonyl C)
- backbones start on the left (the end closer to the head group, or to the first heteroatom of the core) and are
  stacked from top to bottom by their distance to it, tetraether chains span from one polyol to the other
- all other atoms are placed by minimizing the deviations from the distances of RDKit's own depiction for atoms up to
  3 bonds apart (bond lengths, angles and ring shapes) and a repulsion between all other atoms, the backbones stay
  fixed
- without straight chains, RDKit's depiction is rotated by PCA so that its main axis is horizontal with the head group
  on the left
"""
import math
from dataclasses import dataclass

import numpy as np
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Mol, RWMol, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Geometry import Point3D

from LipidCalculator.compound_groups.intact_polar_lipids.generate_ipl import PIECE_NAME_PROP, PIECE_ROLE_PROP

BOND: float = 1.5  # bond length of RDKit depictions
# vertical distance between stacked chains
CHAIN_SPACING: float = 2.2 * BOND
# shortest backbone that is straightened
MIN_BACKBONE: int = 6
# bond length in pixels for drawings with a fixed scale
FIXED_BOND_PIXELS: float = 22
_COS30: float = math.cos(math.radians(30))


@dataclass(frozen=True)
class DepictionOptions:
    straight_chains: bool = True
    hide_heads: bool = False
    # number of chain carbons drawn (from the attachment), 0 draws the whole chains
    shorten_chains: int = 0
    fixed_scale: bool = False


def _small_ring(atom: Chem.Atom) -> bool:
    return any(atom.IsInRingSize(k) for k in range(3, 9))


def _head_label(piece: str) -> str:
    from LipidCalculator.nomenclature.archlips import HEAD_MAP_REVERSE
    return HEAD_MAP_REVERSE.get(piece, piece)


def collapse_heads(mol: Mol) -> Mol:
    """Replace each head group by one atom labeled with its name"""
    rw = RWMol(mol)
    for role in ('head1', 'head2'):
        atoms = [a.GetIdx() for a in rw.GetAtoms() if a.HasProp(PIECE_ROLE_PROP) and a.GetProp(PIECE_ROLE_PROP) == role]
        if not atoms:
            continue
        name = rw.GetAtomWithIdx(atoms[0]).GetProp(PIECE_NAME_PROP)
        anchors = {nb.GetIdx() for idx in atoms for nb in rw.GetAtomWithIdx(idx).GetNeighbors() if nb.GetIdx() not in atoms}
        label = rw.AddAtom(Chem.Atom(0))
        rw.GetAtomWithIdx(label).SetProp('atomLabel', _head_label(name))
        rw.GetAtomWithIdx(label).SetProp(PIECE_ROLE_PROP, role)
        for anchor in anchors:
            rw.AddBond(anchor, label, Chem.BondType.SINGLE)
        for idx in sorted(atoms, reverse=True):
            rw.RemoveAtom(idx)
    out = rw.GetMol()
    out.UpdatePropertyCache(strict=False)
    return out


def _reference_atom(mol: Mol) -> int:
    """Atom that defines the left side: the head group, the first attachment point of building blocks (Fr), otherwise
    the first heteroatom"""
    for role in ('head1', 'head2'):
        for a in mol.GetAtoms():
            if a.HasProp(PIECE_ROLE_PROP) and a.GetProp(PIECE_ROLE_PROP) == role:
                return a.GetIdx()
    placeholders = sorted((a.GetAtomMapNum(), a.GetIdx()) for a in mol.GetAtoms() if a.GetSymbol() == 'Fr')
    if placeholders:
        return placeholders[0][1]
    return next((a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() not in (1, 6)), 0)


def _chain_carbons(mol: Mol) -> set[int]:
    """Carbons of chains: bonded only to carbons or terminal substituents (OH of hydroxylated chains), not aromatic,
    in rings only if the whole ring consists of such carbons (cyclopentane rings of tetraethers, not quinones)"""
    out = {a.GetIdx() for a in mol.GetAtoms()
           if a.GetAtomicNum() == 6 and not a.GetIsAromatic()
           and all(nb.GetAtomicNum() == 6 or (nb.GetDegree() == 1 and nb.GetAtomicNum() > 0) for nb in a.GetNeighbors())}
    for ring in mol.GetRingInfo().AtomRings():
        if len(ring) <= 8 and not out.issuperset(ring):
            out -= set(ring)
    return out


def _neighbors(mol: Mol, atoms: set[int]) -> dict[int, list[int]]:
    return {i: [nb.GetIdx() for nb in mol.GetAtomWithIdx(i).GetNeighbors() if nb.GetIdx() in atoms] for i in atoms}


def _longest_path(mol: Mol, atoms: set[int]) -> list[int]:
    neighbors = _neighbors(mol, atoms)
    starts = [i for i in atoms if len(neighbors[i]) <= 1] or list(atoms)
    best: list[int] = []
    for start in starts:
        stack = [(start, [start])]
        while stack:
            idx, path = stack.pop()
            if len(path) > len(best):
                best = path
            stack.extend((nb, path + [nb]) for nb in neighbors[idx] if nb not in path)
    return best


def _geodesic(neighbors: dict[int, list[int]], a: int, b: int) -> list[int] | None:
    """Shortest path from a to b"""
    previous = {a: None}
    queue = [a]
    while queue:
        idx = queue.pop(0)
        if idx == b:
            path = [b]
            while previous[path[-1]] is not None:
                path.append(previous[path[-1]])
            return path[::-1]
        for nb in neighbors[idx]:
            if nb not in previous:
                previous[nb] = idx
                queue.append(nb)
    return None


def _chain_paths(mol: Mol, component: set[int]) -> list[list[int]]:
    """Chains bonded to the rest of the molecule at both ends (biphytanes) are the shortest paths between their
    attachments, paired so that the paths are shortest in total (single paths over the bridge of GMGTs can be shorter
    than a chain, but force a longer path for the other chain), otherwise the longest path is the chain"""
    neighbors = _neighbors(mol, component)
    # bonded to the rest of the molecule, not to a terminal substituent like OH
    anchors = sorted(i for i in component if any(nb.GetIdx() not in component and nb.GetDegree() > 1
                                                 for nb in mol.GetAtomWithIdx(i).GetNeighbors()))
    geodesics = {(a, b): p for k, a in enumerate(anchors) for b in anchors[k + 1:]
                 if (p := _geodesic(neighbors, a, b)) and len(p) >= MIN_BACKBONE}

    def pairings(free: list[int]) -> list[list[tuple[int, int]]]:
        """All sets of disjoint pairs of anchors with a path"""
        if len(free) < 2:
            return [[]]
        first, rest = free[0], free[1:]
        out = pairings(rest)  # first stays unpaired
        for partner in rest:
            if (first, partner) in geodesics:
                out += [[(first, partner)] + p for p in pairings([a for a in rest if a != partner])]
        return out

    best = min(pairings(anchors), key=lambda p: (-len(p), sum(len(geodesics[pair]) for pair in p)))
    paths = [geodesics[pair] for pair in best]
    if not paths:
        path = _longest_path(mol, component)
        if len(path) >= MIN_BACKBONE:
            paths.append(path)
    return paths


def _over_rings(mol: Mol, path: list[int]) -> list[int]:
    """Lead the path over the short side of the rings it passes (the bridge of the cyclopentane and cyclohexane rings
    of tetraethers), the other ring atoms are drawn as a bulge"""
    for ring in mol.GetRingInfo().AtomRings():
        if len(ring) > 8:
            continue
        inside = [i for i, idx in enumerate(path) if idx in ring]
        if len(inside) < 3 or inside != list(range(inside[0], inside[-1] + 1)):
            continue
        a, b = ring.index(path[inside[0]]), ring.index(path[inside[-1]])
        arcs = [[ring[(a + k) % len(ring)] for k in range((b - a) % len(ring) + 1)],
                [ring[(a - k) % len(ring)] for k in range((a - b) % len(ring) + 1)]]
        path = path[:inside[0]] + min(arcs, key=len) + path[inside[-1] + 1:]
    return path


def find_backbones(mol: Mol) -> list[list[int]]:
    """Backbones of the chains, oriented from the left (see module docstring), in the order they are stacked"""
    hydrocarbon = _chain_carbons(mol)
    seen: set[int] = set()
    paths = []
    for start in sorted(hydrocarbon):
        if start in seen:
            continue
        component, stack = {start}, [start]
        while stack:
            for nb in mol.GetAtomWithIdx(stack.pop()).GetNeighbors():
                if nb.GetIdx() in hydrocarbon and nb.GetIdx() not in component:
                    component.add(nb.GetIdx())
                    stack.append(nb.GetIdx())
        seen |= component
        # fused ring systems (sterols, carotenoid end groups) are not chains
        if sum(_small_ring(mol.GetAtomWithIdx(i)) for i in component) > len(component) / 2:
            continue
        paths += [_over_rings(mol, path) for path in _chain_paths(mol, component)]
    backbones = []
    for path in paths:
        # extend towards the rest of the molecule (C1 and the ether/ester O)
        for end in (0, -1):
            for _ in range(2):
                tip = mol.GetAtomWithIdx(path[end])
                options = [nb for nb in tip.GetNeighbors()
                           if nb.GetIdx() not in path and nb.GetIdx() not in hydrocarbon and not _small_ring(nb)
                           and nb.GetDegree() >= 2]
                if not options:
                    break
                path = [options[0].GetIdx()] + path if end == 0 else path + [options[0].GetIdx()]
        backbones.append(path)
    if not backbones:
        return []
    distances = Chem.GetDistanceMatrix(mol)
    ref = _reference_atom(mol)
    backbones = [p if distances[ref][p[0]] <= distances[ref][p[-1]] else p[::-1] for p in backbones]
    backbones.sort(key=lambda p: (distances[ref][p[0]], p[0]))
    # chains bonded at both ends start at the same polyol as the first one
    spanning = [p for p in backbones if _bonded_at_both_ends(mol, p)]
    if spanning:
        left = spanning[0][0]
        backbones = [p[::-1] if p in spanning[1:] and distances[left][p[-1]] < distances[left][p[0]] else p
                     for p in backbones]
    return backbones


def _default_coords(mol: Mol) -> np.ndarray:
    m = Chem.Mol(mol)
    rdDepictor.Compute2DCoords(m)
    return np.array(m.GetConformer().GetPositions())[:, :2]


def _pca_orient(mol: Mol, coords: np.ndarray) -> np.ndarray:
    """Rotate the main axis horizontally, the reference atom (head group) to the left"""
    centered = coords - coords.mean(axis=0)
    _, vectors = np.linalg.eigh(np.cov(centered.T))
    main = vectors[:, -1]
    rotation = np.array([[main[0], main[1]], [-main[1], main[0]]])
    out = centered @ rotation.T
    if out[_reference_atom(mol), 0] > 0:
        out[:, 0] *= -1
    return out


def _rotation(angle: float) -> np.ndarray:
    return np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])


def _overlap(points: np.ndarray, others: np.ndarray) -> float:
    if not len(points) or not len(others):
        return 0.0
    dist = np.linalg.norm(points[:, None] - others[None], axis=-1)
    return float((np.clip(1.8 * BOND - dist, 0, None) ** 2).sum())


def _rigid_placement(mol: Mol, fixed: dict[int, np.ndarray], ideal: np.ndarray) -> np.ndarray:
    """Initial positions: each connected part of free atoms (polyols with head groups, ring bulges, branches) as a rigid
    copy of RDKit's depiction, rotated onto its fixed neighbors (mirrored if this overlaps less with what is placed)"""
    n = mol.GetNumAtoms()
    pos = ideal.copy()
    placed = np.zeros(n, dtype=bool)
    for idx, xy in fixed.items():
        pos[idx], placed[idx] = xy, True
    free = set(range(n)) - set(fixed)
    neighbors = _neighbors(mol, free)
    parts = []
    for start in sorted(free):
        if any(start in part for part in parts):
            continue
        part, stack = [start], [start]
        while stack:
            for nb in neighbors[stack.pop()]:
                if nb not in part:
                    part.append(nb)
                    stack.append(nb)
        parts.append(part)
    # big parts first, they are placed most reliably
    for part in sorted(parts, key=len, reverse=True):
        anchors = sorted({nb.GetIdx() for i in part for nb in mol.GetAtomWithIdx(i).GetNeighbors()
                          if nb.GetIdx() in fixed})
        if not anchors:
            continue
        candidates = []
        for mirror in (1, -1):
            x = ideal[part] * [1, mirror]
            a_ideal = ideal[anchors] * [1, mirror]
            if len(anchors) >= 2:
                # rotation that maps the anchors of the ideal depiction onto the fixed ones (Kabsch)
                ca, cb = a_ideal.mean(axis=0), pos[anchors].mean(axis=0)
                u, _, vt = np.linalg.svd((a_ideal - ca).T @ (pos[anchors] - cb))
                d = np.sign(np.linalg.det(vt.T @ u.T)) or 1.0
                rotation = vt.T @ np.diag([1, d]) @ u.T
                candidates.append((x - ca) @ rotation.T + cb)
            else:
                # point the part away from the placed neighbors of its anchor, or to the sides if this is taken
                # (several branches at one carbon, e.g. the methyl and hydroxyl group at C3)
                a = anchors[0]
                around = [nb.GetIdx() for nb in mol.GetAtomWithIdx(a).GetNeighbors() if placed[nb.GetIdx()]]
                outward = pos[a] - pos[around].mean(axis=0) if around else np.array([0.0, 1.0])
                inward = x.mean(axis=0) - a_ideal[0]
                angle = math.atan2(outward[1], outward[0]) - math.atan2(inward[1], inward[0])
                for turn in (0, 30, -30, 60, -60, 90, -90, 120, -120, 150, -150, 180):
                    candidates.append((x - a_ideal[0]) @ _rotation(angle + math.radians(turn)).T + pos[a])
        others = pos[placed]
        best = min(candidates, key=lambda c: _overlap(c, others))
        pos[part], placed[part] = best, True
    return pos


def _spring_layout(mol: Mol, fixed: dict[int, np.ndarray], iterations: int = 400) -> np.ndarray:
    """Place the free atoms (see module docstring)"""
    n = mol.GetNumAtoms()
    ideal = _default_coords(mol)
    graph = Chem.GetDistanceMatrix(mol)
    if not fixed:
        return ideal
    pos = _rigid_placement(mol, fixed, ideal)
    free = np.array([i not in fixed for i in range(n)])
    if not free.any():
        return pos
    # target distances of nearby atoms from RDKit's depiction, repulsion of all other pairs
    near = (graph > 0) & (graph <= 3)
    target = np.where(near, np.linalg.norm(ideal[:, None] - ideal[None], axis=-1), 0.0)
    repel = (graph > 3) | (graph == 0) & ~np.eye(n, dtype=bool)
    # atoms on top of each other (RDKit's depiction of crowded macrocycles can have them) have no direction to
    # separate, a slight offset gives them one
    pos[free] += np.random.default_rng(0).normal(scale=0.01 * BOND, size=(int(free.sum()), 2))
    step = 0.1
    for it in range(iterations):
        diff = pos[:, None] - pos[None]
        dist = np.maximum(np.linalg.norm(diff, axis=-1), 1e-6) + np.eye(n)
        unit = diff / dist[..., None]
        # springs between nearby atoms
        force = -((dist - target) * near)[..., None] * unit
        # repulsion closer than 1.8 bonds
        overlap = np.clip(1.8 * BOND - dist, 0, None) * repel
        force += 0.5 * overlap[..., None] * unit
        total = force.sum(axis=1)
        total[~free] = 0
        pos += step * total
        if it == iterations // 2:
            step = 0.05
    if not np.isfinite(pos).all():
        # should not happen, but a broken layout must not break the drawing
        return _default_coords(mol)
    return pos


def _bonded_at_both_ends(mol: Mol, path: list[int]) -> bool:
    return all(mol.GetAtomWithIdx(path[end]).GetAtomicNum() != 6 for end in (0, -1))


def _straight_layout(mol: Mol, backbones: list[list[int]]) -> np.ndarray:
    """Backbones as horizontal zigzag lines: chains bonded at both ends (tetraethers) are stretched to the same length
    so that their polyols line up, chains bonded to the far polyol (phytanyl chains of GTGTs) start on the right"""
    step = BOND * _COS30
    spanning = [p for p in backbones if _bonded_at_both_ends(mol, p)]
    width = max((len(p) - 1 for p in spanning), default=0) * step
    distances = Chem.GetDistanceMatrix(mol)
    ref = _reference_atom(mol)
    fixed: dict[int, np.ndarray] = {}
    for k, path in enumerate(backbones):
        if path in spanning:
            x = np.linspace(0, width, len(path))
        elif spanning and distances[ref][path[0]] > (len(spanning[0]) - 1) / 2:
            x = width - np.arange(len(path)) * step
        else:
            x = np.arange(len(path)) * step
        for i, idx in enumerate(path):
            if idx not in fixed:
                fixed[idx] = np.array([x[i], -k * CHAIN_SPACING + (BOND / 2 if i % 2 else 0.0)])
    return _spring_layout(mol, fixed)


def shorten_chains(mol: Mol, coords: np.ndarray, backbones: list[list[int]], n_keep: int,
                   straight: bool) -> tuple[Mol, np.ndarray]:
    """Draw only the first n_keep carbons of each backbone (and the last ones of backbones bonded at both ends,
    e.g. biphytanes), the removed atoms are replaced by a label with their sum formula"""
    removed: set[int] = set()
    windows: list[tuple[float, float]] = []
    chain_carbons = _chain_carbons(mol)
    for path in backbones:
        # extension atoms towards the rest of the molecule are not counted
        head = next(i for i, idx in enumerate(path) if idx in chain_carbons)
        tail = next(i for i, idx in reversed(list(enumerate(path))) if idx in chain_carbons)
        bonded_right = tail < len(path) - 1
        stop = tail + 1 - n_keep if bonded_right else len(path)
        window = path[head + n_keep:stop]
        if len(window) < 2:
            continue
        removed |= set(window)
        windows.append((coords[window[0], 0], coords[window[-1], 0]))
    if not removed:
        return mol, coords
    # branches of the removed backbone atoms (methyl groups, rings) go with them
    kept_backbone = {idx for path in backbones for idx in path} - removed
    stack = list(removed)
    while stack:
        for nb in mol.GetAtomWithIdx(stack.pop()).GetNeighbors():
            j = nb.GetIdx()
            if j not in removed and j not in kept_backbone and nb.GetAtomicNum() == 6:
                removed.add(j)
                stack.append(j)
    rw = RWMol(mol)
    coords = list(map(np.array, coords))
    # connected parts of the removed atoms each become one label
    parts: list[set[int]] = []
    for idx in sorted(removed):
        if any(idx in part for part in parts):
            continue
        part, stack = {idx}, [idx]
        while stack:
            for nb in mol.GetAtomWithIdx(stack.pop()).GetNeighbors():
                if nb.GetIdx() in removed and nb.GetIdx() not in part:
                    part.add(nb.GetIdx())
                    stack.append(nb.GetIdx())
        parts.append(part)
    # close the gap left by removed middle parts (straight chains of tetraethers)
    shift = 0.0
    if straight and windows:
        left, right = min(w[0] for w in windows), max(w[1] for w in windows)
        right_of = [i for i in range(len(coords)) if i not in removed and coords[i][0] > right + 0.1]
        if right_of:
            shift = (right - left) - 2 * BOND * _COS30
            for i in right_of:
                coords[i] = coords[i] - np.array([shift, 0.0])
    for part in parts:
        n_c = sum(mol.GetAtomWithIdx(i).GetAtomicNum() == 6 for i in part)
        n_h = sum(mol.GetAtomWithIdx(i).GetTotalNumHs() for i in part)
        boundary = {nb.GetIdx() for i in part for nb in mol.GetAtomWithIdx(i).GetNeighbors() if nb.GetIdx() not in removed}
        label = rw.AddAtom(Chem.Atom(0))
        rw.GetAtomWithIdx(label).SetProp('atomLabel', f'C<sub>{n_c}</sub>H<sub>{n_h}</sub>')
        for b in boundary:
            rw.AddBond(b, label, Chem.BondType.SINGLE)
        first = min(part, key=lambda i: coords[i][0])
        xy = np.mean([coords[b] for b in boundary], axis=0) if len(boundary) > 1 else \
            coords[min(boundary)] + (coords[first] - coords[min(boundary)]) * 1.5
        coords.append(xy)
    keep = [i for i in range(len(coords)) if i not in removed]
    for idx in sorted(removed, reverse=True):
        rw.RemoveAtom(idx)
    out = rw.GetMol()
    out.UpdatePropertyCache(strict=False)
    return out, np.array([coords[i] for i in keep])


def depict(mol: Mol, options: DepictionOptions = DepictionOptions()) -> Mol:
    """Copy of the molecule with 2D coordinates (and labels) of the depiction"""
    m = Chem.RemoveHs(mol)
    # explicit H counts left from building make RDKit label carbons (C instead of a vertex)
    for atom in m.GetAtoms():
        if atom.GetAtomicNum() == 6 and not atom.GetFormalCharge() and not atom.GetNumRadicalElectrons():
            atom.SetNumExplicitHs(0)
            atom.SetNoImplicit(False)
    m.UpdatePropertyCache(strict=False)
    if options.hide_heads:
        m = collapse_heads(m)
    backbones = find_backbones(m) if options.straight_chains or options.shorten_chains else []
    if options.straight_chains and backbones:
        coords = _straight_layout(m, backbones)
    else:
        coords = _pca_orient(m, _default_coords(m))
    if options.shorten_chains and backbones:
        m, coords = shorten_chains(m, coords, backbones, options.shorten_chains, options.straight_chains)
    conformer = Chem.Conformer(m.GetNumAtoms())
    for i, (x, y) in enumerate(coords):
        conformer.SetAtomPosition(i, Point3D(float(x), float(y), 0.0))
    m.RemoveAllConformers()
    m.AddConformer(conformer, assignId=True)
    return m


def draw(mol: Mol, options: DepictionOptions = DepictionOptions(), size: tuple[int, int] = (1200, 675)) -> Image.Image:
    """PNG image (transparent background) of the depiction"""
    import io
    drawer = rdMolDraw2D.MolDraw2DCairo(*size)
    draw_options = drawer.drawOptions()
    draw_options.clearBackground = False
    if options.fixed_scale:
        draw_options.fixedBondLength = FIXED_BOND_PIXELS
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, depict(mol, options))
    drawer.FinishDrawing()
    return Image.open(io.BytesIO(drawer.GetDrawingText()))
