"""
Modify the isoprenoid chains of a lipid: number of double bonds, chain length and removal of chains.

CONVENTIONS
===========
- chains are given in the notation of e.g. Archaeol(20:1_25:3(Me)), so carbon atoms and double bonds
  for each chain, optionally followed by a modification (OH or Me)
- chains of the core that already carry a modification are identified by it, all other chains by their order
- for cores whose chains are counted in isoprene units (e.g. quinones, MK(7:7)) the numbers are units
- a chain with 0 carbon atoms is removed (the attachment atom keeps an H, e.g. lyso archaeols)
- a modification on a chain without one is added: (OH) to the first branched carbon (C3 of archaeols,
  e.g. MK(4:2(OH)) where the first isoprene unit is hydrated), (Me) two carbons further (C5)
- the isoprene units of a chain are its methyl branched backbone carbons with the regular spacing of 4
- double bonds are placed from the attachment site onwards between each branched carbon and the
  preceding backbone carbon (Δ2, Δ6, Δ10, ... like in geranylgeranyl), chains are elongated or
  shortened by isoprene units at the ω-end
- the chains of a molecule are identified either
    - explicitly by atom map numbers on the first carbon of each chain (order = map number) or
    - automatically as acyclic carbon chains bound to an ether oxygen of a glycerol unit, ordered
      by glycerol unit, sn-3 (primary carbon) before sn-2 (secondary carbon)
- tetraethers (GDGTs, ...) get double bonds in the two biphytanyl half chains at their first glycerol unit,
  alternating between the sn-2 and the sn-3 half chain (e.g. 3 double bonds: Δ2 and Δ6 in the sn-2 half,
  Δ2 in the sn-3 half), see add_tetraether_double_bonds
"""
import re
from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import Mol, RWMol

ISOPRENE_CARBONS: int = 5
# a chain must have at least this many carbons to be considered in the automatic detection
MIN_CHAIN_CARBONS: int = 10
MIN_TETRAETHER_HALF_CARBONS: int = 3
# glycerol, butanetriol, pentanetriol, ...
MAX_POLYOL_CARBONS: int = 6
# rings up to this size are real rings (cyclopentanes, sugars, aromatics), bigger ones are the macrocycles of
# tetraethers
MAX_RING_SIZE: int = 8


def _in_small_ring(atom: Chem.Atom) -> bool:
    return any(atom.IsInRingSize(n) for n in range(3, MAX_RING_SIZE + 1))


@dataclass
class ChainSpec:
    n_units: int
    n_double_bonds: int
    modification: str | None = None  # 'OH' or 'Me'


def _unit_positions(mol: Mol, backbone: list[int], atoms: set[int]) -> list[int]:
    """Indices in the backbone of the methyl branched carbons that define the isoprene units"""
    on_backbone = set(backbone)
    branched = [j for j, idx in enumerate(backbone)
                if any(nb.GetIdx() in atoms and nb.GetIdx() not in on_backbone
                       for nb in mol.GetAtomWithIdx(idx).GetNeighbors())]
    # first branch after the attachment atom determines the phase, additional methyls (Me) are ignored
    first = next((j for j in branched if j >= 1), None)
    if first is None:
        return []
    return [j for j in branched if (j - first) % 4 == 0]


@dataclass
class IsoprenoidChain:
    atoms: set[int]  # all carbon atoms of the chain (including methyl branches)
    backbone: list[int]  # longest path, starting at the atom bonded to the rest of the molecule
    modification: str | None
    n_units: int


def parse_chain_spec(spec: str | list[tuple[int, int]], in_units: bool = False) -> list[ChainSpec]:
    """'20:1_25:0(OH)' --> [ChainSpec(4, 1), ChainSpec(5, 0, 'OH')]"""
    if not isinstance(spec, str):
        return [ChainSpec(n // (1 if in_units else ISOPRENE_CARBONS), db) for n, db in spec]
    spec = spec.strip()
    # optional outer parentheses, e.g. '(20:1_25:0(OH))'
    if spec.startswith('(') and spec.endswith(')'):
        spec = spec[1:-1]
    out = []
    for part in spec.split('_'):
        match = re.fullmatch(r'C?(\d+):(\d+)(?:\((OH|Me)\))?', part.strip())
        if match is None:
            raise ValueError(f'chain "{part}" must have the form 20:1, 25:0(OH) or 20:2(Me)')
        n, db, mod = int(match.group(1)), int(match.group(2)), match.group(3)
        if not in_units and n % ISOPRENE_CARBONS != 0:
            raise ValueError(f'isoprenoid chains must have a multiple of 5 carbon atoms, got {n}')
        out.append(ChainSpec(n if in_units else n // ISOPRENE_CARBONS, db, mod))
    return out


def _carbon_component(mol: Mol, start: int, exclude: set[int]) -> set[int]:
    """All carbons reachable from start via carbon atoms that are not in rings"""
    seen, stack = {start}, [start]
    while stack:
        atom = mol.GetAtomWithIdx(stack.pop())
        for nb in atom.GetNeighbors():
            idx = nb.GetIdx()
            if idx in seen or idx in exclude or nb.GetSymbol() != 'C' or _in_small_ring(nb):
                continue
            seen.add(idx)
            stack.append(idx)
    return seen


def _longest_path(mol: Mol, start: int, atoms: set[int]) -> list[int]:
    best: list[int] = [start]
    stack = [(start, [start])]
    while stack:
        idx, path = stack.pop()
        if len(path) > len(best):
            best = path
        for nb in mol.GetAtomWithIdx(idx).GetNeighbors():
            if nb.GetIdx() in atoms and nb.GetIdx() not in path:
                stack.append((nb.GetIdx(), path + [nb.GetIdx()]))
    return best


def _make_chain(mol: Mol, start: int, anchor: int) -> IsoprenoidChain | None:
    atoms = _carbon_component(mol, start, exclude={anchor})
    if any(_in_small_ring(mol.GetAtomWithIdx(a)) for a in atoms):
        return None
    backbone = _longest_path(mol, start, atoms)
    has_oh = any(nb.GetSymbol() == 'O' and nb.GetDegree() == 1
                 for a in atoms for nb in mol.GetAtomWithIdx(a).GetNeighbors())
    n_units = len(_unit_positions(mol, backbone, atoms))
    has_me = len(atoms) > n_units * ISOPRENE_CARBONS
    return IsoprenoidChain(atoms, backbone, 'OH' if has_oh else ('Me' if has_me else None), n_units)


def find_isoprenoid_chains(mol: Mol) -> list[IsoprenoidChain]:
    """Chains in the order in which they are specified"""
    # explicitly marked chains
    marked = sorted((a.GetAtomMapNum(), a.GetIdx()) for a in mol.GetAtoms()
                    if a.GetAtomMapNum() > 0 and a.GetSymbol() == 'C')
    if marked:
        chains = []
        for _, idx in marked:
            atom = mol.GetAtomWithIdx(idx)
            anchors = [nb.GetIdx() for nb in atom.GetNeighbors() if nb.GetSymbol() != 'C' or _in_small_ring(nb)]
            assert len(anchors) == 1, f'could not determine attachment of marked chain at atom {idx}'
            chain = _make_chain(mol, idx, anchors[0])
            assert chain is not None, f'marked chain at atom {idx} is not acyclic'
            chains.append(chain)
        return chains

    # automatic detection: ether O between glycerol carbon and chain
    found = [(key, chain) for key, chain, is_tetraether in _ether_chains(mol) if not is_tetraether]
    found.sort(key=lambda t: t[0])
    return [chain for _, chain in found]


def _ether_chains(mol: Mol):
    """Yields (glycerol unit, is sn-2), chain and whether the chain is (half of) a biphytane for all long
    acyclic carbon chains bound to an ether oxygen of a glycerol unit"""
    for o in mol.GetAtoms():
        if o.GetSymbol() != 'O' or o.GetDegree() != 2 or _in_small_ring(o):
            continue
        c1, c2 = [nb for nb in o.GetNeighbors()]
        for chain_start, glycerol in ((c1, c2), (c2, c1)):
            if chain_start.GetSymbol() != 'C' or glycerol.GetSymbol() != 'C':
                continue
            chain = _make_chain(mol, chain_start.GetIdx(), o.GetIdx())
            if chain is None:
                continue
            # chains of tetraethers (biphytanes) are bound to two ether oxygens or end in a ring
            n_ethers = sum(nb.GetSymbol() == 'O' and nb.GetDegree() == 2
                           for a in chain.atoms for nb in mol.GetAtomWithIdx(a).GetNeighbors())
            touches_ring = any(_in_small_ring(nb) for a in chain.atoms for nb in mol.GetAtomWithIdx(a).GetNeighbors())
            is_tetraether = n_ethers != 1 or touches_ring
            # half chains of tetraethers can be short if they end in a ring, but need an isoprene unit
            if len(chain.atoms) < (MIN_TETRAETHER_HALF_CARBONS if is_tetraether else MIN_CHAIN_CARBONS) \
                    or (is_tetraether and chain.n_units == 0):
                continue
            # the other side has to be a small polyol (glycerol, butanetriol, ...), not another chain
            polyol = _carbon_component(mol, glycerol.GetIdx(), exclude={o.GetIdx()})
            if len(polyol) > MAX_POLYOL_CARBONS:
                continue
            yield (min(polyol), glycerol.GetDegree() > 2), chain, is_tetraether


def has_biphytanyl_chains(mol: Mol) -> bool:
    """Whether double bonds can be added with add_tetraether_double_bonds"""
    # tetraethers have biphytanyl chains at two glycerol units or form a macrocycle (GDGTs with many rings
    # may have no half chains)
    tetraether_units = {unit for (unit, _), _, is_tetraether in _ether_chains(mol) if is_tetraether}
    return len(tetraether_units) >= 2 or any(len(ring) > MAX_RING_SIZE for ring in mol.GetRingInfo().AtomRings())


def add_tetraether_double_bonds(mol: Mol, n_double_bonds: int) -> Mol:
    """Double bonds in the biphytanyl half chains of the first glycerol unit, if they do not fit there, also in those
    of the second one (see module docstring)"""
    units_to_halves = {}
    for (unit, is_secondary), chain, is_tetraether in _ether_chains(mol):
        if is_tetraether:
            units_to_halves.setdefault(unit, []).append((not is_secondary, chain))
    if not has_biphytanyl_chains(mol):
        raise ValueError('molecule has no biphytanyl chains')
    # cores with many rings may have no half chains, then only the inner segments are used
    rw = RWMol(mol)
    order = []
    for unit in sorted(units_to_halves):
        candidates = []
        # sn-2 half first
        for _, chain in sorted(units_to_halves[unit], key=lambda t: t[0]):
            units = [j for j in _unit_positions(rw, chain.backbone, chain.atoms) if j >= 1]
            # only the units of this half (a biphytane has 4 units per half)
            candidates.append([(chain.backbone[j - 1], chain.backbone[j]) for j in units[:4]])
        # alternate between the half chains
        order += [c[k] for k in range(max(map(len, candidates))) for c in candidates if k < len(c)]
    # if this is not enough (cores with many rings): next to any methyl branch in the inner segments
    for b in rw.GetAtoms():
        if b.GetSymbol() != 'C' or _in_small_ring(b) or not any(
                nb.GetSymbol() == 'C' and nb.GetDegree() == 1 for nb in b.GetNeighbors()):
            continue
        order += [(a.GetIdx(), b.GetIdx()) for a in b.GetNeighbors()
                  if a.GetSymbol() == 'C' and a.GetDegree() == 2 and not _in_small_ring(a)]
    placed = 0
    for a, b in order:
        if placed == n_double_bonds:
            break
        has_double = any(bond.GetBondType() == Chem.BondType.DOUBLE
                         for idx in (a, b) for bond in rw.GetAtomWithIdx(idx).GetBonds())
        if not has_double and _num_hs(rw, a) and _num_hs(rw, b):
            rw.GetBondBetweenAtoms(a, b).SetBondType(Chem.BondType.DOUBLE)
            for idx in (a, b):
                rw.GetAtomWithIdx(idx).SetNumExplicitHs(0)
                rw.GetAtomWithIdx(idx).SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
            placed += 1
    if placed < n_double_bonds:
        raise ValueError(f'could only place {placed} of {n_double_bonds} double bonds in the biphytanyl chains')
    out = rw.GetMol()
    Chem.SanitizeMol(out)
    return out


def describe_isoprenoid_chains(mol: Mol, in_units: bool = False) -> str:
    """e.g. '20:0_20:0(OH)', empty string if the molecule has no isoprenoid chains"""
    parts = []
    for chain in find_isoprenoid_chains(mol):
        n_db = sum(mol.GetBondBetweenAtoms(a, b).GetBondTypeAsDouble() == 2
                   for a, b in zip(chain.backbone[:-1], chain.backbone[1:]))
        n = chain.n_units if in_units else chain.n_units * ISOPRENE_CARBONS
        parts.append(f'{n}:{n_db}' + (f'({chain.modification})' if chain.modification else ''))
    return '_'.join(parts)


def _match_specs(chains: list[IsoprenoidChain], specs: list[ChainSpec]) -> list[tuple[IsoprenoidChain, ChainSpec]]:
    if len(chains) != len(specs):
        raise ValueError(f'molecule has {len(chains)} isoprenoid chain(s), but {len(specs)} were specified')
    assigned: dict[int, int] = {}  # index of spec -> index of chain
    # chains with a modification are identified by it
    for i, spec in enumerate(specs):
        matches = [c for c, chain in enumerate(chains)
                   if spec.modification and chain.modification == spec.modification and c not in assigned.values()]
        if matches:
            assigned[i] = matches[0]
    # all other chains by their order, modifications are added to them
    free = [c for c in range(len(chains)) if c not in assigned.values()]
    for i, spec in enumerate(specs):
        if i in assigned:
            continue
        c = free.pop(0)
        if chains[c].modification is not None:
            raise ValueError(f'chain {c + 1} of the core has the modification {chains[c].modification}, '
                             f'specify it like 20:0({chains[c].modification})')
        assigned[i] = c
    return [(chains[assigned[i]], spec) for i, spec in enumerate(specs)]


def _num_hs(rw: RWMol, idx: int) -> int:
    atom = rw.GetAtomWithIdx(idx)
    atom.UpdatePropertyCache(strict=False)
    return atom.GetTotalNumHs()


def modify_isoprenoid_chains(mol: Mol, specs: str | list[ChainSpec], in_units: bool = False) -> Mol:
    """Set length and number of double bonds of all isoprenoid chains in mol (see module docstring)"""
    if isinstance(specs, str):
        specs = parse_chain_spec(specs, in_units=in_units)
    chains = find_isoprenoid_chains(mol)
    if not chains:
        raise ValueError('molecule has no isoprenoid chains')
    rw = RWMol(mol)
    to_remove: set[int] = set()
    for chain, spec in _match_specs(chains, specs):
        backbone = list(chain.backbone)
        if spec.n_units == 0:
            to_remove |= chain.atoms
            continue
        # remove existing double bonds in the backbone
        for a, b in zip(backbone[:-1], backbone[1:]):
            rw.GetBondBetweenAtoms(a, b).SetBondType(Chem.BondType.SINGLE)
        delta = spec.n_units - chain.n_units
        atoms = set(chain.atoms)
        # elongate at the ω-end: terminal C - C - C - C(CH3) - C
        for _ in range(max(delta, 0)):
            prev = backbone[-1]
            rw.GetAtomWithIdx(prev).SetNumExplicitHs(0)
            for k in range(4):
                new = rw.AddAtom(Chem.Atom(6))
                rw.AddBond(prev, new, Chem.BondType.SINGLE)
                backbone.append(new)
                atoms.add(new)
                if k == 2:
                    methyl = rw.AddAtom(Chem.Atom(6))
                    rw.AddBond(new, methyl, Chem.BondType.SINGLE)
                    atoms.add(methyl)
                prev = new
        # shorten: remove the last unit(s) of the backbone with their branches
        for _ in range(max(-delta, 0)):
            unit, backbone = backbone[-4:], backbone[:-4]
            for idx in unit:
                to_remove.add(idx)
                to_remove |= {nb.GetIdx() for nb in rw.GetAtomWithIdx(idx).GetNeighbors()
                              if nb.GetIdx() not in backbone and nb.GetSymbol() == 'C'}
        units = _unit_positions(rw, backbone, atoms - to_remove)
        if spec.modification and chain.modification is None:
            # first branched carbon after the attachment atom (C3 for archaeols)
            first = next(j for j in units if j >= 1)
            position, element = (first, 8) if spec.modification == 'OH' else (first + 2, 6)
            atom = rw.GetAtomWithIdx(backbone[position])
            atom.SetNumExplicitHs(0)
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
            rw.AddBond(atom.GetIdx(), rw.AddAtom(Chem.Atom(element)), Chem.BondType.SINGLE)
        # double bonds between each branched carbon and the preceding backbone carbon (Δ2, Δ6, ...)
        placed = 0
        for j in units:
            if placed == spec.n_double_bonds:
                break
            if j < 1:
                continue
            a, b = backbone[j - 1], backbone[j]
            if _num_hs(rw, a) > 0 and _num_hs(rw, b) > 0:
                rw.GetBondBetweenAtoms(a, b).SetBondType(Chem.BondType.DOUBLE)
                for idx in (a, b):
                    atom = rw.GetAtomWithIdx(idx)
                    atom.SetNumExplicitHs(0)
                    atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
                placed += 1
        if placed < spec.n_double_bonds:
            raise ValueError(f'could only place {placed} of {spec.n_double_bonds} double bonds in chain with '
                             f'{spec.n_units} isoprene units')
    for idx in sorted(to_remove, reverse=True):
        rw.RemoveAtom(idx)
    out = rw.GetMol()
    # bond orders changed, so let rdkit recompute the H counts of the carbons
    for atom in out.GetAtoms():
        if atom.GetSymbol() == 'C' and not atom.GetIsAromatic():
            atom.SetNoImplicit(False)
    Chem.SanitizeMol(out)
    return out


def clear_chain_marks(mol: Mol) -> Mol:
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return mol


# ---------- modifications of ether-bound isoprenoid chains (hydroxyl and methyl groups) ----------
# C3 of isoprenoid chains bound to an ether O (ether O, C1, C2, C3 with a methyl branch or ring), the polyol side does
# not match because its carbon next to C1 is not a CH2, linear chains do not match because their C3 is not branched
_C3_PATTERN = Chem.MolFromSmarts('[OD2]-[CH2]-[CH2]-[CX4;!H0;D3]')
# C13 of a biphytanyl half: C11(Me), C12, C13, C14, C15(Me), C16 and C16' (center), C15'(Me)
_C13_PATTERN = Chem.MolFromSmarts('[CX4](-[CH3])-[CH2]-[CH2;!r5;!r6]-[CH2]-[CX4](-[CH3])-[CH2]-[CH2]-[CX4]-[CH3]')
# C13 of the half of H-shaped tetraethers (GMGTs) whose C15 methyl is the bridge to the other chain
_C13_CROSSLINKED_PATTERN = Chem.MolFromSmarts('[CX4](-[CH3])-[CH2]-[CH2;!r5;!r6]-[CH2]-[CH1;!r5;!r6](-[CH2]-[CH1])')


def _half_rank(mol: Mol, atom_idx: int, distances) -> tuple[int, int]:
    """Order of the half chains: sn-2 (ether O at a secondary polyol carbon) before sn-3, then by polyol (atom map
    number of its placeholder, the order of the head groups)"""
    ethers = [a for a in mol.GetAtoms() if a.GetSymbol() == 'O' and a.GetDegree() == 2 and not _in_small_ring(a)]
    o = min(ethers, key=lambda e: distances[atom_idx][e.GetIdx()])
    polyol = [nb for nb in o.GetNeighbors() if distances[atom_idx][nb.GetIdx()] > distances[atom_idx][o.GetIdx()]]
    if not polyol:
        return 2, 0
    carbon = polyol[0]
    secondary = carbon.GetTotalNumHs() == 1
    # placeholder of the polyol (sn-1 of glycerol) within 3 bonds
    maps = [a.GetAtomMapNum() for a in mol.GetAtoms()
            if a.GetSymbol() == 'Fr' and distances[carbon.GetIdx()][a.GetIdx()] <= 3]
    return 0 if secondary else 1, min(maps, default=0)


_HYDROXY_SITES: tuple[list[Mol], int] = ([_C3_PATTERN], 3)
_METHYL_SITES: tuple[list[Mol], int] = ([_C13_PATTERN, _C13_CROSSLINKED_PATTERN], 3)


def _sites(mol: Mol, patterns: list[Mol], position: int) -> set[int]:
    return {match[position] for pattern in patterns for match in mol.GetSubstructMatches(pattern)}


def n_hydroxy_positions(mol: Mol) -> int:
    """Number of hydroxyl groups add_isoprenoid_hydroxy can add"""
    return len(_sites(mol, *_HYDROXY_SITES))


def n_methyl_positions(mol: Mol) -> int:
    """Number of methyl groups add_isoprenoid_methyls can add"""
    return len(_sites(mol, *_METHYL_SITES))


def _add_groups(mol: Mol, n: int, patterns: list[Mol], position: int, element: int, name: str) -> Mol:
    """Add n atoms of element (as OH or CH3) at the atom at position of the matches of patterns"""
    if not n:
        return mol
    distances = Chem.GetDistanceMatrix(mol)
    candidates = sorted(_sites(mol, patterns, position), key=lambda idx: (_half_rank(mol, idx, distances), idx))
    if len(candidates) < n:
        raise ValueError(f'could only place {len(candidates)} of {n} {name}')
    rw = RWMol(mol)
    for idx in candidates[:n]:
        atom = rw.GetAtomWithIdx(idx)
        atom.SetNumExplicitHs(0)
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        rw.AddBond(idx, rw.AddAtom(Chem.Atom(element)), Chem.BondType.SINGLE)
    out = rw.GetMol()
    Chem.SanitizeMol(out)
    return out


def add_isoprenoid_hydroxy(mol: Mol, n: int) -> Mol:
    """Hydroxyl groups at C3 of the ether-bound isoprenoid chains (hydroxyarchaeols, OH-GDGTs, ...), sn-2 chains
    first, then sn-3 chains, in the order of the polyols"""
    return _add_groups(mol, n, *_HYDROXY_SITES, 8, 'hydroxyl group(s)')


def add_isoprenoid_methyls(mol: Mol, n: int) -> Mol:
    """Additional methyl groups at C13 of the biphytanyl half chains (Homo-Me-GDGTs, ...), sn-2 halves first, then
    sn-3 halves, in the order of the polyols"""
    return _add_groups(mol, n, *_METHYL_SITES, 6, 'methyl group(s)')
