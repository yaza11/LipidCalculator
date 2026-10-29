"""
LIPID MAPS shorthand notation (Liebisch et al., J. Lipid Res. 2020, doi:10.1194/jlr.S120001025), which is also used by
RefMet and MS-DIAL, the standard for bacterial and eukaryotic lipids:
- glycero(phospho)lipids: <class> <sum composition> (species level, e.g. PC 34:1) or <class> <chain>_<chain>
  (molecular species, e.g. PC 16:0_18:1), ether lipids with O- (PC O-16:0_18:1, species PC O-34:1), dialkyl lipids
  with dO- at the species level (archaeol: DG dO-40:0, hydroxyarchaeol: DG dO-40:0;O)
- sphingolipids: <class> <sphingoid base>;O2/<N-acyl chain>, e.g. Cer 18:1;O2/16:0
"""
import re

from LipidCalculator.compound_groups.intact_polar_lipids.isoprenoid_blocks import is_block_core, parse_block_core
from LipidCalculator.compound_groups.intact_polar_lipids.pieces_from_json import ABBREVIATION_TO_GROUP, CORE_GROUPS
from LipidCalculator.nomenclature.lipid import Lipid, NotSupported

# MS-DIAL class -> head groups on DAG/AEG (ether lipids), heads with chains (Acyl, APG, ...) take further chains
MSDIAL_GLYCERO: dict[str, list[str]] = {
    'PC': ['PC'], 'PE': ['PE'], 'PS': ['PS'], 'PG': ['PG'], 'PI': ['PI'], 'PA': ['P'], 'PMeOH': ['PMeOH'],
    'PEtOH': ['PEth'], 'MGDG': ['1G'], 'DGDG': ['2G'], 'SQDG': ['SQ'], 'DGTS': ['TMHS'], 'DGGA': ['GluA'],
    'DGCC': ['DGCC'], 'DG': [], 'LPC': ['PC'], 'LPE': ['PE'], 'LPS': ['PS'], 'LPG': ['PG'], 'LPI': ['PI'],
    'LPA': ['P'], 'LDGTS': ['TMHS'], 'LDGCC': ['DGCC'], 'MG': [], 'TG': ['Acyl'], 'HBMP': ['APG'], 'BMP': ['APG'],
    'ADGGA': ['AGluA'], 'CL': ['PGPA'],
}
# lyso lipids get an empty chain at sn-2
MSDIAL_LYSO: set[str] = {'LPC', 'LPE', 'LPS', 'LPG', 'LPI', 'LPA', 'LDGTS', 'LDGCC', 'MG'}
# MS-DIAL class -> head groups on the ceramide core
MSDIAL_SPHINGO: dict[str, list[str]] = {
    'Cer': [], 'SM': ['PC'], 'HexCer': ['1G'], 'Hex2Cer': ['2G'], 'Hex3Cer': ['3G'], 'SHexCer': ['Sulfo-Gly'],
    'CerP': ['P'], 'GM3': ['GM3'], 'AHexCer': ['AHex'], 'PI-Cer': ['PI'], 'SPB': [], 'SL': [],
}
# MS-DIAL class with one or two chains on a fixed skeleton -> core
MSDIAL_ACYL: dict[str, str] = {'CE': 'CE', 'CAR': 'CAR', 'VAE': 'VAE', 'NAE': 'NAE', 'NAGly': 'NAGly',
                               'NAGlySer': 'NAGlySer', 'NAOrn': 'OL'}
CHAIN: str = r'\d+:\d+(?:;(?:\(\d+OH\)|\d*O))*'

# head groups (single) -> class of glycero(phospho)lipids and sphingolipids
GLYCERO_CLASS: dict[str | None, str] = {
    None: 'DG', 'PC': 'PC', 'PE': 'PE', 'PS': 'PS', 'PG': 'PG', 'PI': 'PI', 'P': 'PA', 'PMeOH': 'PMeOH',
    'PEth': 'PEtOH', '1G': 'MGDG', '2G': 'DGDG', 'SQ': 'SQDG', 'TMHS': 'DGTS', 'GluA': 'DGGA', 'DGCC': 'DGCC',
}
LYSO_CLASS: dict[str, str] = {'DG': 'MG', 'PC': 'LPC', 'PE': 'LPE', 'PS': 'LPS', 'PG': 'LPG', 'PI': 'LPI',
                              'PA': 'LPA', 'DGTS': 'LDGTS', 'DGCC': 'LDGCC'}
SPHINGO_CLASS: dict[str | None, str] = {None: 'Cer', 'PC': 'SM', '1G': 'HexCer', '2G': 'Hex2Cer', '3G': 'Hex3Cer',
                                        'Sulfo-Gly': 'SHexCer', 'P': 'CerP', 'GM3': 'GM3', 'PI': 'PI-Cer'}
SPHINGO_CORES: dict[str, str] = {'CER': 'O2', 'DHCER': 'O2', 'PCER': 'O3'}


def sphingoid_core(n_oxygens: int, double_bonds: int, cls: str) -> str:
    """Core for a sphingoid base with n_oxygens (2O: sphingosine/sphinganine, 3O: phytosphingosine)"""
    if cls == 'SL':
        return 'SL'
    if n_oxygens == 2:
        # sphinganine (no 4E double bond) based lipids use the dihydroceramide core
        return 'DHCER' if double_bonds == 0 else 'CER'
    if n_oxygens == 3:
        return 'PCER'
    raise NotSupported(f'sphingoid base with {n_oxygens} O')


def split_sphingoid_sum(n: int, double_bonds: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Sum composition of sphingoid base and N-acyl chain: any valid split gives the same formula, 18 carbons (less
    for short totals) for the sphingoid base with at most one double bond, the remaining double bonds in the N-acyl
    chain if they fit"""
    n_sph = 18 if n >= 20 else max(6, n - 2)
    n_acyl = n - n_sph
    db_sph = min(double_bonds, 1)
    db_acyl = min(double_bonds - db_sph, max(n_acyl - 2, 0))
    return (n_sph, double_bonds - db_acyl), (n_acyl, db_acyl)


def lipid_from_pieces(names: list[str], kwargs: dict, family: str) -> Lipid:
    """Heads before the core, chains after it"""
    groups = [ABBREVIATION_TO_GROUP.get(n) for n in names]
    i_core = next(i for i, g in enumerate(groups) if g in CORE_GROUPS)
    return Lipid(tuple(names[:i_core]), names[i_core], tuple(names[i_core + 1:]),
                 core_chains=kwargs.get('core_chains'), core_double_bonds=kwargs.get('core_double_bonds', 0),
                 family=family)


def parse(name: str) -> Lipid:
    """Building blocks for names in LIPID MAPS/MS-DIAL notation, e.g. PC 16:0_18:1, Cer 18:1;2O/16:0;(2OH),
    TG 16:0_18:1_18:2"""
    if m := re.fullmatch(r'CoQ(\d+)', name):
        return Lipid(core='UQ', core_chains=f'{m[1]}:{m[1]}', family='MS-DIAL CoQ')
    m = re.fullmatch(r'(?P<cls>[A-Za-z0-9-]+) (?P<rest>.+)', name.strip())
    if m is None:
        raise NotSupported('unknown lipid type')
    cls, rest = m['cls'], m['rest']
    if '(d' in rest:
        raise NotSupported('isotope labelled standard')
    family = f'MS-DIAL {cls}'
    if cls in MSDIAL_GLYCERO:
        rest = re.sub(r'-SN\d$', '', rest)
        if m := re.fullmatch(r'dO-(\d+):(\d+)(?:;O(\d?))?', rest):
            # dialkyl glycerols: isoprenoid ones are archaeols (the LIPID MAPS notation for archaeol)
            if (int(m[1]), int(m[2])) != (40, 0):
                return Lipid(tuple(MSDIAL_GLYCERO[cls]), 'DEG', (f'C{m[1]}:{m[2]}',), family=family)
            n_oh = int(m[3] or 1) if m.group(0).count(';O') else 0
            return Lipid(tuple(MSDIAL_GLYCERO[cls]), 'AR',
                         core_chains=['20:0_20:0', '20:0(OH)_20:0', '20:0(OH)_20:0(OH)'][n_oh], family='archaeol')
        if not re.fullmatch(r'(?:[OP]-)?\d+:\d+(?:_\d+:\d+)*', rest):
            raise NotSupported(f'unknown {cls} notation')
        ether = rest.startswith(('O-', 'P-'))
        chains = [list(map(int, c.split(':'))) for c in (rest[2:] if ether else rest).split('_')]
        if rest.startswith('P-'):
            # plasmalogen: vinyl ether, so one more double bond in the ether chain
            chains[0][1] += 1
        if cls == 'CL':
            # CL 34:1_36:2: compositions of the two phosphatidyl units, each split into two chains
            chains = [c for n, db in chains for c in ([n // 2, db // 2], [n - n // 2, db - db // 2])]
        names = [f'C{n}:{db}' for n, db in chains]
        if cls in MSDIAL_LYSO:
            names += ['C0:0']
        elif cls == 'BMP':
            # one acyl chain on each glycerol, sn-1 of the core glycerol is free
            names = ['C0:0'] + names
        return Lipid(tuple(MSDIAL_GLYCERO[cls]), 'AEG' if ether else 'DAG', tuple(names), family=family)
    if cls in MSDIAL_SPHINGO:
        heads = list(MSDIAL_SPHINGO[cls])
        extra = []
        # acyl chain on the head group (AHexCer) or at the 3-OH (acyl sphingomyelin)
        if a := re.match(r'\(O-(' + CHAIN + r')\)', rest):
            extra, rest = ['C' + a[1]], rest[a.end():]
        if a := re.search(r'\(FA (' + CHAIN + r')\)$', rest):
            extra, rest = ['C' + a[1]], rest[:a.start()]
            heads.append('Acyl')
        s = re.fullmatch(r'(\d+):(\d+);(\d?)O(\d?)(?:/(' + CHAIN + r'))?', rest)
        if s is None:
            raise NotSupported(f'unknown {cls} notation')
        # both MS-DIAL (18:1;2O) and LIPID MAPS (18:1;O2) notation
        n, db, n_oxygens = int(s[1]), int(s[2]), int(s[3] or s[4] or 1)
        if cls == 'SPB':
            acyl = 'C0:0'
        elif s[5]:
            acyl = 'C' + s[5]
        else:
            (n, db), (n_acyl, db_acyl) = split_sphingoid_sum(n, db)
            acyl = f'C{n_acyl}:{db_acyl}'
        core = sphingoid_core(n_oxygens, db, cls)
        return Lipid(tuple(heads), core, (f'C{n}:{db}', acyl, *extra), family=family)
    if cls in MSDIAL_ACYL:
        if not re.fullmatch(CHAIN + r'(?:/' + CHAIN + r')?', rest):
            raise NotSupported(f'unknown {cls} notation')
        return Lipid(core=MSDIAL_ACYL[cls], chains=tuple('C' + c for c in rest.split('/')), family=family)
    if cls in ('SE', 'ST'):
        # sterol (ester) skeletons are cores named like the database without the acyl chain
        if s := re.fullmatch(r'(?P<key>.+?)(?:/|;FA )(?P<chain>' + CHAIN + r')', name):
            return Lipid(core=s['key'], chains=('C' + s['chain'],), family=family)
        return Lipid(core=name, family=family)
    raise NotSupported(f'MS-DIAL class {cls}')


# ---------- formatting ----------
def _chain(chain: str) -> str:
    """'C16:0;(2OH)' --> '16:0;(2OH)'"""
    return chain[1:]


def _sum(chains: tuple[str, ...]) -> tuple[int, int]:
    n = db = 0
    for chain in chains:
        m = re.match(r'C(\d+):(\d+)', chain)
        n, db = n + int(m[1]), db + int(m[2])
    return n, db


def _archaeol_dialkyl(lipid: Lipid) -> str | None:
    """Sum composition of archaeol cores in LIPID MAPS notation, e.g. dO-40:0;O, None if it cannot be written"""
    from LipidCalculator.nomenclature.archlips import _archaeol_chains
    chains = _archaeol_chains(lipid)
    if chains is None or '(Me)' in chains:
        return None
    n = db = n_oh = 0
    for chain in chains.split('_'):
        m = re.fullmatch(r'(\d+):(\d+)(\(OH\))?', chain)
        if m is None:
            return None
        n, db, n_oh = n + int(m[1]), db + int(m[2]), n_oh + bool(m[3])
    return f'dO-{n}:{db}' + ({0: '', 1: ';O'}.get(n_oh, f';O{n_oh}'))


def name(lipid: Lipid) -> str | None:
    """LIPID MAPS shorthand at the level the building blocks define (sum composition for a single chain, molecular
    species for two chains), None for lipids without LIPID MAPS shorthand"""
    if len(lipid.heads) > 1:
        return None
    head = lipid.heads[0] if lipid.heads else None
    core, chains = lipid.core, lipid.chains
    if core in ('DAG', 'AEG', 'DEG') or core == 'AR' or (is_block_core(core) and parse_block_core(core)[0] == 'AR'):
        if head not in GLYCERO_CLASS:
            return None
        cls = GLYCERO_CLASS[head]
        if core not in ('DAG', 'AEG', 'DEG'):
            composition = _archaeol_dialkyl(lipid)
            return None if composition is None else f'{cls} {composition}'
        if 'C0:0' in chains:
            if cls not in LYSO_CLASS:
                return None
            cls, chains = LYSO_CLASS[cls], tuple(c for c in chains if c != 'C0:0')
            if core == 'AEG' and lipid.chains[0] == 'C0:0':
                core = 'DAG'  # the ether chain is empty
        if not chains:
            return None
        if len(chains) == 1:
            n, db = _sum(chains)
            prefix = {'DAG': '', 'AEG': 'O-', 'DEG': 'dO-' if len(lipid.chains) == 1 else 'O-'}[core]
            return f'{cls} {prefix}{n}:{db}'
        prefixes = {'DAG': ('', ''), 'AEG': ('O-', ''), 'DEG': ('O-', 'O-')}[core]
        return f'{cls} ' + '_'.join(p + _chain(c) for p, c in zip(prefixes, chains))
    if core in SPHINGO_CORES:
        if head not in SPHINGO_CLASS or len(chains) != 2:
            return None
        sph, acyl = chains
        return f'{SPHINGO_CLASS[head]} {_chain(sph)};{SPHINGO_CORES[core]}/{_chain(acyl)}'
    return None
