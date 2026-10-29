"""
Build isoprenoid cores (tetraethers, diethers and hydrocarbons) from isoprene units.

CONVENTIONS
===========
- cores are given as <type>(<chain>_<chain>_...), e.g. GDGT(40:0_40:0(2cp)), AR(20:0_25:0) or HT(19:0)
- the type defines the polyols and how the chains are bonded to them (see CORE_TYPES):
    - tetraethers: GDGT, BDGT, PDGT (glycerol, butanetriol or pentanetriol - glycerol, two chains), GMGT (H-shaped
      GDGT, the chains are bonded to each other at their centers), GTGT (one biphytanyl chain, one phytanyl chain at
      each glycerol), GDD, BDD, PDD (polyol, two chains ending in OH), BPD (one chain with OH at both ends)
    - diethers: AR (archaeol, chain 1 at sn-3, chain 2 at sn-2), MAR (macrocyclic archaeol)
    - hydrocarbons: HT (regular, head-to-tail, e.g. phytane HT(20:0), pristane HT(19:0)), HH (head-to-head, e.g.
      biphytane HH(40:0)), TT (tail-to-tail, e.g. squalane TT(30:0), lycopane TT(40:0), crocetane TT(20:0), PMI
      TT(25:0))
- chains consist of halves (phytanyl-like parts made of isoprene units), positions in a half are counted from its
  C1 (head, bonded to the ether O), methyl branches are at C3, C7, C11, C15, ... and the ω-end is an isopropyl group
- chain kinds:
    - bridge: two halves joined at their ω-ends (head-to-head, biphytanyl), the half at the first anchor gets the
      extra unit for odd numbers of units
    - single: one half with a free ω-end (phytanyl in archaeol)
    - tail-to-tail: two halves joined at their C1 (squalane), the first half gets the extra unit
- chains are given as <C atoms>:<double bonds>, optionally followed by modifications in parentheses, separated by
  commas and with an optional count: cp (cyclopentane ring), ch (cyclohexane ring), OH (hydroxyl group),
  e.g. 40:0, 40:1(2cp,OH), 25:0. The number of C atoms must be a multiple of 5 (isoprene units), only regular
  hydrocarbons (HT) can be shortened at their head (pristane, norpristane, ...)
- cyclopentane rings bond the methyl of Cn to Cn+3, they are placed at C7 of the first, C7 of the second half,
  then C3 (first, second half), C11, ... (as in GDGT-1 to GDGT-8)
- cyclohexane rings bond the methyl of C11 to C15 (as in crenarchaeol), first half first, they are placed before
  the cyclopentane rings
- hydroxyl groups go to C3 of the first, C3 of the second half, then C7, ... (also in rings, as long as the carbon
  has an H)
- double bonds go between each branched carbon and the preceding backbone carbon (Δ2, Δ6, ...), alternating
  between the halves
- tetraethers with two bridges are antiparallel: chain 1 is bonded to the first O of polyol 1 (sn-3 of glycerol)
  and the second O of polyol 2 (sn-2), chain 2 the other way round (as in GDGT-0)
- the cross-link of GMGTs bonds the methyl of C15 of chain 2 to C14 of chain 1 (both in the half at glycerol 1), as
  in the MS-DIAL archaeal lipid library (see GMGT_CROSSLINK)
- head groups bond to the free O of the polyols (sn-1 of glycerol, C1 of butanetriol and pentanetriol), polyol 1
  first, then to the free OH groups of the chains (chain 1 first). These placeholders are marked by atom map numbers.
"""
import math
import re
from dataclasses import dataclass, field

from rdkit import Chem
from rdkit.Chem import Mol, RWMol

ISOPRENE_CARBONS: int = 5
# backbone carbons per isoprene unit (the fifth carbon is the methyl branch)
UNIT_BACKBONE: int = 4

# anchor of a chain end: (polyol, O of the polyol), 'OH' for a free hydroxyl group, None for a hydrocarbon end
Anchor = tuple[int, int] | str | None


@dataclass(frozen=True)
class ChainLayout:
    kind: str  # 'bridge', 'single' or 'tail-to-tail'
    anchors: tuple[Anchor, ...] = ()
    description: str = ''


@dataclass(frozen=True)
class CrossLink:
    """Bond between the methyl of a branched carbon of one chain and a backbone carbon of another chain, both given
    as (chain, half, position)"""
    methyl: tuple[int, int, int]
    backbone: tuple[int, int, int]


@dataclass(frozen=True)
class CoreType:
    description: str
    polyols: tuple[str, ...]
    chains: tuple[ChainLayout, ...]
    crosslink: CrossLink | None = None


# H-shaped GDGTs as in the MS-DIAL archaeal lipid library (library_arch_mp1_msdial_jul.sqlite): the methyl of C15 of
# the first half of chain 2 is bonded to C14 of the first half of chain 1 (reproduces 11 of its 16 GMGT cores, the
# others are positional isomers)
GMGT_CROSSLINK = CrossLink(methyl=(1, 0, 15), backbone=(0, 0, 14))


def _tetraether(polyol: str, description: str, crosslink: CrossLink | None = None) -> CoreType:
    # glycerol - glycerol antiparallel, C2 of butanetriol/pentanetriol with the primary O of glycerol (as in the
    # archaeal lipid library)
    other = (1, 1) if polyol == 'glycerol' else (1, 0)
    return CoreType(description, (polyol, 'glycerol'), (
        ChainLayout('bridge', ((0, 0), other), 'biphytanyl'),
        ChainLayout('bridge', ((0, 1), (1, 1 - other[1])), 'biphytanyl'),
    ), crosslink)


def _diol(polyol: str, description: str) -> CoreType:
    return CoreType(description, (polyol,), (
        ChainLayout('bridge', ((0, 0), 'OH'), 'biphytanyl'),
        ChainLayout('bridge', ((0, 1), 'OH'), 'biphytanyl'),
    ))


CORE_TYPES: dict[str, CoreType] = {
    'GDGT': _tetraether('glycerol', 'glycerol dibiphytanyl glycerol tetraether'),
    'BDGT': _tetraether('butanetriol', 'butanetriol dibiphytanyl glycerol tetraether'),
    'PDGT': _tetraether('pentanetriol', 'pentanetriol dibiphytanyl glycerol tetraether'),
    'GMGT': _tetraether('glycerol', 'glycerol monoalkyl glycerol tetraether (H-shaped)', crosslink=GMGT_CROSSLINK),
    'GTGT': CoreType('glycerol trialkyl glycerol tetraether', ('glycerol', 'glycerol'), (
        ChainLayout('bridge', ((0, 0), (1, 1)), 'biphytanyl'),
        ChainLayout('single', ((0, 1),), 'phytanyl at sn-2 of glycerol 1'),
        ChainLayout('single', ((1, 0),), 'phytanyl at sn-3 of glycerol 2'),
    )),
    'GDD': _diol('glycerol', 'glycerol dibiphytanyl diol'),
    'BDD': _diol('butanetriol', 'butanetriol dibiphytanyl diol'),
    'PDD': _diol('pentanetriol', 'pentanetriol dibiphytanyl diol'),
    'BPD': CoreType('biphytanyl diol', (), (ChainLayout('bridge', ('OH', 'OH'), 'biphytanyl'),)),
    'AR': CoreType('archaeol (diether)', ('glycerol',), (
        ChainLayout('single', ((0, 0),), 'phytanyl at sn-3'),
        ChainLayout('single', ((0, 1),), 'phytanyl at sn-2'),
    )),
    'MAR': CoreType('macrocyclic archaeol', ('glycerol',), (ChainLayout('bridge', ((0, 0), (0, 1)), 'biphytanyl'),)),
    'HT': CoreType('regular isoprenoid alkane (head-to-tail, e.g. phytane, pristane)', (), (
        ChainLayout('single', (None,), 'head-to-tail'),)),
    'HH': CoreType('head-to-head isoprenoid alkane (e.g. biphytane)', (), (
        ChainLayout('bridge', (None, None), 'head-to-head'),)),
    'TT': CoreType('tail-to-tail isoprenoid alkane (e.g. squalane, lycopane, crocetane, PMI)', (), (
        ChainLayout('tail-to-tail', (), 'tail-to-tail'),)),
}
# typical length (C atoms) of a chain of the kind
DEFAULT_CARBONS: dict[str, int] = {'bridge': 40, 'single': 20, 'tail-to-tail': 30}
# hydrocarbon chains that may be shortened at their head (e.g. pristane)
_TRUNCATABLE: set[str] = {'HT'}
# head carbons that are removed in this order when shortening a half, positions or ('methyl', position)
_TRUNCATION_ORDER: list[int | tuple[str, int]] = [1, 2, ('methyl', 3), 3]

BLOCK_CORE_RE = re.compile(rf'({"|".join(sorted(CORE_TYPES, key=len, reverse=True))})\((.+)\)')
_CHAIN_RE = re.compile(r'C?(\d+):(\d+)(?:\(([^()]*)\))?')
_MODIFICATION_RE = re.compile(r'(\d*)(cp|ch|OH)')


@dataclass
class ChainBlock:
    n_carbons: int = 40
    n_double_bonds: int = 0
    n_cyclopentanes: int = 0
    n_cyclohexanes: int = 0
    n_hydroxy: int = 0

    def __str__(self) -> str:
        mods = [f'{n if n > 1 else ""}{mod}' for n, mod in ((self.n_cyclopentanes, 'cp'),
                                                           (self.n_cyclohexanes, 'ch'),
                                                           (self.n_hydroxy, 'OH')) if n]
        return f'{self.n_carbons}:{self.n_double_bonds}' + (f'({",".join(mods)})' if mods else '')


def is_block_core(name: str) -> bool:
    return BLOCK_CORE_RE.fullmatch(name) is not None


def format_block_core(core_type: str, chains: list[ChainBlock]) -> str:
    """e.g. GDGT(40:0_40:0(2cp))"""
    return f'{core_type}({"_".join(map(str, chains))})'


def _parse_chain(chain: str) -> ChainBlock:
    match = _CHAIN_RE.fullmatch(chain.strip())
    if match is None:
        raise ValueError(f'chain "{chain}" must have the form 40:0 or 40:1(2cp,OH)')
    out = ChainBlock(n_carbons=int(match.group(1)), n_double_bonds=int(match.group(2)))
    mods = match.group(3)
    for mod in (mods.split(',') if mods else []):
        match_mod = _MODIFICATION_RE.fullmatch(mod.strip())
        if match_mod is None:
            raise ValueError(f'modification "{mod}" must be cp, ch or OH with an optional count, e.g. 2cp')
        count = int(match_mod.group(1) or 1)
        attr = {'cp': 'n_cyclopentanes', 'ch': 'n_cyclohexanes', 'OH': 'n_hydroxy'}[match_mod.group(2)]
        setattr(out, attr, getattr(out, attr) + count)
    return out


def parse_block_core(spec: str) -> tuple[str, list[ChainBlock]]:
    """'GDGT(40:0_40:0(2cp))' --> ('GDGT', [ChainBlock(40), ChainBlock(40, n_cyclopentanes=2)])"""
    match = BLOCK_CORE_RE.fullmatch(spec.strip())
    if match is None:
        raise ValueError(f'"{spec}" is not an isoprenoid core, use e.g. GDGT(40:0_40:0(2cp)) with one of the types '
                         f'{", ".join(CORE_TYPES)}')
    core_type = match.group(1)
    chains = [_parse_chain(chain) for chain in match.group(2).split('_')]
    n_chains = len(CORE_TYPES[core_type].chains)
    if len(chains) != n_chains:
        raise ValueError(f'{core_type} has {n_chains} chain(s), got {len(chains)}')
    for chain in chains:
        if chain.n_carbons % ISOPRENE_CARBONS and core_type not in _TRUNCATABLE:
            raise ValueError(f'isoprenoid chains of {core_type} must have a multiple of 5 carbon atoms, '
                             f'got {chain.n_carbons}')
    return core_type, chains


@dataclass
class _Half:
    positions: dict[int, int] = field(default_factory=dict)  # position (counted from C1) --> atom
    methyls: dict[int, int] = field(default_factory=dict)  # position of the branched carbon --> methyl carbon

    @property
    def head(self) -> int:
        return self.positions[min(self.positions)]

    @property
    def tail(self) -> int:
        return self.positions[max(self.positions)]

    def segment(self, start: int, stop: int) -> list[int] | None:
        """Atoms at positions start to stop, None if one of them does not exist"""
        if not all(p in self.positions for p in range(start, stop + 1)):
            return None
        return [self.positions[p] for p in range(start, stop + 1)]


def _carbon(rw: RWMol) -> int:
    return rw.AddAtom(Chem.Atom(6))


def _add_half(rw: RWMol, n_units: int, n_removed: int = 0) -> _Half:
    """Half with n_units isoprene units, the first n_removed carbons (see _TRUNCATION_ORDER) are left out"""
    removed = _TRUNCATION_ORDER[:n_removed]
    half, previous = _Half(), None
    for position in range(1, n_units * UNIT_BACKBONE + 1):
        if position in removed:
            continue
        atom = _carbon(rw)
        if previous is not None:
            rw.AddBond(previous, atom, Chem.BondType.SINGLE)
        half.positions[position] = previous = atom
        if position % UNIT_BACKBONE == 3 and ('methyl', position) not in removed:
            half.methyls[position] = _carbon(rw)
            rw.AddBond(atom, half.methyls[position], Chem.BondType.SINGLE)
    return half


def _place_rings(rw: RWMol, halves: list[_Half], chain: ChainBlock, in_ring: set[int]) -> None:
    """Rings close between the methyl of a branched carbon and a backbone carbon 3 (cyclopentane) or 4
    (cyclohexane) positions further"""
    n_positions = max(halves[0].positions)
    for n_rings, span, first_positions, name in ((chain.n_cyclohexanes, 4, [11], 'cyclohexane'),
                                                 (chain.n_cyclopentanes, 3, [7, 3], 'cyclopentane')):
        positions = first_positions + [p for p in range(11, n_positions, UNIT_BACKBONE) if p not in first_positions]
        placed = 0
        for half, p in ((half, p) for p in positions for half in halves):
            if placed == n_rings:
                break
            segment = half.segment(p, p + span)
            if segment is None or p not in half.methyls:
                continue
            ring = [half.methyls[p]] + segment
            if in_ring & set(ring):
                continue
            rw.AddBond(half.methyls[p], segment[-1], Chem.BondType.SINGLE)
            in_ring.update(ring)
            placed += 1
        if placed < n_rings:
            raise ValueError(f'could only place {placed} of {n_rings} {name} ring(s) in a chain with '
                             f'{chain.n_carbons} carbon atoms')


def _branched_positions(halves: list[_Half]) -> list[tuple[_Half, int]]:
    """Branched positions alternating between the halves (C3 of the first, C3 of the second half, C7, ...)"""
    positions = sorted({p for half in halves for p in half.methyls})
    return [(half, p) for p in positions for half in halves if p in half.methyls]


def _place_hydroxy(rw: RWMol, halves: list[_Half], chain: ChainBlock) -> None:
    placed = 0
    for half, p in _branched_positions(halves):
        if placed == chain.n_hydroxy:
            break
        atom = half.positions.get(p)
        if atom is None or rw.GetAtomWithIdx(atom).GetDegree() >= 4:
            continue
        rw.AddBond(atom, rw.AddAtom(Chem.Atom(8)), Chem.BondType.SINGLE)
        placed += 1
    if placed < chain.n_hydroxy:
        raise ValueError(f'could only place {placed} of {chain.n_hydroxy} hydroxyl group(s) in a chain with '
                         f'{chain.n_carbons} carbon atoms')


def _place_double_bonds(rw: RWMol, halves: list[_Half], chain: ChainBlock, in_ring: set[int]) -> None:
    def free(idx: int) -> bool:
        atom = rw.GetAtomWithIdx(idx)
        return idx not in in_ring and atom.GetDegree() < 4 and not any(
            bond.GetBondType() == Chem.BondType.DOUBLE for bond in atom.GetBonds()) and not any(
            nb.GetSymbol() == 'O' for nb in atom.GetNeighbors())

    placed = 0
    for half, p in _branched_positions(halves):
        if placed == chain.n_double_bonds:
            break
        segment = half.segment(p - 1, p)
        if segment is not None and all(map(free, segment)):
            rw.GetBondBetweenAtoms(*segment).SetBondType(Chem.BondType.DOUBLE)
            placed += 1
    if placed < chain.n_double_bonds:
        raise ValueError(f'could only place {placed} of {chain.n_double_bonds} double bond(s) in a chain with '
                         f'{chain.n_carbons} carbon atoms')


def _add_chain(rw: RWMol, chain: ChainBlock, kind: str) -> list[_Half]:
    """Add a chain with its modifications, the halves are in the order of the anchors"""
    n_units = math.ceil(chain.n_carbons / ISOPRENE_CARBONS)
    n_removed = n_units * ISOPRENE_CARBONS - chain.n_carbons
    min_units = 1 if kind == 'single' else 2
    if n_units < min_units:
        raise ValueError(f'{kind} chains need at least {min_units * ISOPRENE_CARBONS} carbon atoms, '
                         f'got {chain.n_carbons}')
    if kind == 'single':
        halves = [_add_half(rw, n_units, n_removed)]
    else:
        halves = [_add_half(rw, n) for n in ((n_units + 1) // 2, n_units // 2)]
        joined = [half.tail for half in halves] if kind == 'bridge' else [half.head for half in halves]
        rw.AddBond(*joined, Chem.BondType.SINGLE)
    in_ring: set[int] = set()
    _place_rings(rw, halves, chain, in_ring)
    _place_hydroxy(rw, halves, chain)
    _place_double_bonds(rw, halves, chain, in_ring)
    return halves


def _add_placeholder(rw: RWMol, o: int, map_number: int) -> None:
    fr = rw.AddAtom(Chem.Atom(87))
    rw.GetAtomWithIdx(fr).SetAtomMapNum(map_number)
    rw.AddBond(o, fr, Chem.BondType.SINGLE)


def _add_polyol(rw: RWMol, polyol: str, map_number: int) -> list[int]:
    """Add glycerol, butanetriol or pentanetriol with a placeholder at the free O, returns the two O atoms the chains
    bond to"""
    c1, c2, c3 = _carbon(rw), _carbon(rw), _carbon(rw)
    rw.AddBond(c1, c2, Chem.BondType.SINGLE)
    rw.AddBond(c2, c3, Chem.BondType.SINGLE)
    os = []
    for c in (c1, c2, c3):
        os.append(rw.AddAtom(Chem.Atom(8)))
        rw.AddBond(c, os[-1], Chem.BondType.SINGLE)
    _add_placeholder(rw, os[0], map_number)
    if polyol == 'glycerol':
        # sn-1 is free, chains at sn-3 (primary) and sn-2 (secondary)
        return [os[2], os[1]]
    # C1 is free, chains at C2 and C3, C4 (and C5) form a methyl (ethyl) group
    previous = c3
    for _ in range({'butanetriol': 1, 'pentanetriol': 2}[polyol]):
        new = _carbon(rw)
        rw.AddBond(previous, new, Chem.BondType.SINGLE)
        previous = new
    return [os[1], os[2]]


def _add_crosslink(rw: RWMol, chain_halves: list[list[_Half]], crosslink: CrossLink) -> None:
    (c1, h1, p1), (c2, h2, p2) = crosslink.methyl, crosslink.backbone
    methyl = chain_halves[c1][h1].methyls.get(p1)
    backbone = chain_halves[c2][h2].positions.get(p2)
    if methyl is None or backbone is None:
        raise ValueError(f'the chains are too short for the cross-link (methyl of C{p1}, C{p2})')
    for idx in (methyl, backbone):
        atom = rw.GetAtomWithIdx(idx)
        if atom.GetDegree() >= 4 or any(b.GetBondType() == Chem.BondType.DOUBLE for b in atom.GetBonds()):
            raise ValueError(f'the cross-link (methyl of C{p1}, C{p2}) collides with a ring or double bond')
    rw.AddBond(methyl, backbone, Chem.BondType.SINGLE)


def block_core_mol(spec: str) -> Mol:
    """Core with Fr placeholders (marked by atom map numbers in the order in which head groups are attached)"""
    core_type, chains = parse_block_core(spec)
    layout = CORE_TYPES[core_type]
    rw = RWMol()
    polyols = [_add_polyol(rw, polyol, k + 1) for k, polyol in enumerate(layout.polyols)]
    map_number = len(polyols)
    chain_halves = []
    for chain, chain_layout in zip(chains, layout.chains):
        halves = _add_chain(rw, chain, chain_layout.kind)
        chain_halves.append(halves)
        for half, anchor in zip(halves, chain_layout.anchors):
            if anchor == 'OH':
                o = rw.AddAtom(Chem.Atom(8))
                rw.AddBond(half.head, o, Chem.BondType.SINGLE)
                map_number += 1
                _add_placeholder(rw, o, map_number)
            elif anchor is not None:
                polyol, k = anchor
                rw.AddBond(half.head, polyols[polyol][k], Chem.BondType.SINGLE)
    if layout.crosslink is not None:
        _add_crosslink(rw, chain_halves, layout.crosslink)
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    return mol


def block_core_smiles(spec: str) -> str:
    # rdkit writes O bonded to Fr as [O], which would keep no H once the placeholder is removed (all O atoms with
    # brackets are bonded to placeholders, the cores have no charges or radicals)
    return Chem.MolToSmiles(block_core_mol(spec)).replace('[O]', 'O')
