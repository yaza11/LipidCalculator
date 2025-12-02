import itertools
import os
import traceback

from msIO.annotations.main import IonPeak, CompoundGroup, Molecule, Compound, FragmentPeak, SqlBaseClassComp
from rdkit import Chem
from rdkit.Chem import Mol, rdMolDescriptors
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from tqdm import tqdm

from LipidCalculator.rdkit.cleaving import get_fragments
from LipidCalculator.rdkit.adduct.rdkit_add_adduct import add_adduct_to_heteroatom
from LipidCalculator.compound_groups.intact_polar_lipids.generate_ipl import ipl_automatic_bonds
from LipidCalculator.compound_groups.intact_polar_lipids.pieces_from_json import ABBREVIATION_TO_GROUP
from LipidCalculator.compound_groups.to_sql import submit_to_db_inside_session

chain1s = [f'C{i}:0' for i in range(6, 31, 1)] + [f'C{i}:1' for i in range(6, 31, 1)]
chain2s = [f'C{i}:0' for i in range(6, 31, 1)] + [f'C{i}:1' for i in range(6, 31, 1)]
# combine heads, cores, chains in all possible combinations to get characteristic fragments
HEADS = [k for k, v in ABBREVIATION_TO_GROUP.items() if v == 'head groups']
CORES = [k for k, v in ABBREVIATION_TO_GROUP.items() if v == 'core lipids']
HEADS.append(None)


def to_orm_mol(mol: Mol):
    orm = Molecule(
        smiles=Chem.MolToSmiles(mol),
        formula=rdMolDescriptors.CalcMolFormula(mol),
        M=rdMolDescriptors.CalcExactMolWt(mol),
        charge=Chem.GetFormalCharge(mol)
    )
    return orm


def add_mol_as_comp(session, pieces: dict[str, str]):
    def _add_fragment(f, is_loss):
        if f is None:
            return
        f_mol = to_orm_mol(f)
        f_peak = FragmentPeak(
            molecule=f_mol,
            is_fragment=True,
            is_neutral_loss=is_loss,
            mz=f_mol.M / f_mol.charge if not is_loss else None,
            adduct='M+' if not is_loss else None,
        )
        frags.append(f_peak)

    pieces = {k: v for k, v in pieces.items() if v is not None}

    comp_name: str = ' '.join(pieces.values())
    if comp_name in compounds_in_db:
        return

    try:
        groups = [ipl] + [str_to_compound_group_obj[piece] for piece in pieces.values() if
                          piece in str_to_compound_group_obj]

        compound_mol: Mol = ipl_automatic_bonds(
            pieces.values(), split_chain=False, sort_chains=False
        )

        cpd = Compound(
            name=comp_name,
            molecule=to_orm_mol(compound_mol)
        )
        # only H+ adduct for now
        # TODO: other adducts
        mol_with_adduct = to_orm_mol(add_adduct_to_heteroatom(compound_mol))
        ions = [
            IonPeak(adduct='[M+H]+',
                    mz=mol_with_adduct.M / mol_with_adduct.charge,
                    molecule=mol_with_adduct)
        ]

        ion_to_frags = {}
        for ion in ions:
            # TODO: take adduct into account for fragments?
            frags_pos, frags_neut = get_fragments(
                compound_mol, max_recursion_depth=0)
            frags = []
            for f, l in zip(frags_pos, frags_neut):
                _add_fragment(f, False)
                _add_fragment(l, True)
            ion_to_frags[ion] = frags

        submit_to_db_inside_session(
            session=session,
            compound=cpd,
            compound_groups=groups,
            ions=ions,
            frags=ion_to_frags,
        )
        compounds_in_db.add(comp_name)
        with open(log_file_suc, 'a') as f:
            f.write(comp_name + '\n')
    except Exception as e:
        with open(log_file_errs, 'a') as f:
            header = f'Creation for {comp_name} failed: {e}\n'
            tb_str = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
            indented_tb = ''.join('    ' + line for line in tb_str.splitlines(True))
            log_msg = header + indented_tb
            f.write(log_msg)
        print(log_msg)


restart = True

# folder = r'\\hlabstorage.dmz.marum.de\scratch\Yannick\compounds\LipidCalculator'
folder = r"C:\Users\Yannick Zander\Downloads"
db_path = os.path.join(folder, 'database.db')

log_file_errs = os.path.join(folder, 'errors.log')
log_file_suc = os.path.join(folder, 'created.log')

if restart:
    if os.path.exists(db_path):
        os.remove(db_path)
    if os.path.exists(log_file_errs):
        os.remove(log_file_errs)
    if os.path.exists(log_file_suc):
        os.remove(log_file_suc)
    compounds_in_db = set()
else:
    # get which compounds were already created and skip this in the loop
    with open(log_file_suc, 'r') as f:
        compounds_in_db = set(f.readlines())

engine = create_engine(f"sqlite+pysqlite:///{db_path}", future=True)

SqlBaseClassComp.metadata.create_all(engine)

# TODO: use json directly
# start by submitting compound groups to the database
if restart:
    str_to_compound_group_obj = {s: CompoundGroup(name=s if s is not None else '', abbreviation=s) for s in
                                 HEADS + CORES}
    ipl = CompoundGroup(name='intact polar lipids', abbreviation='IPL')
    with Session(engine) as session:
        for g in str_to_compound_group_obj.values():
            ipl.children.append(g)
        session.add(ipl)
        session.commit()
else:
    # TODO: fetch ipl and other compound groups from file
    str_to_compound_group_obj = {s: CompoundGroup(name=s, abbreviation=s) for s in HEADS + CORES}
    ipl = CompoundGroup(name='intact polar lipids', abbreviation='IPL')
    raise NotImplementedError

mols: dict[str, Mol] = {}
pred_ms2: dict[str, dict[float, str | None]] = {}
pred_losses: dict[str, dict[float, str | None]] = {}

heads_cores = list(itertools.product(HEADS, CORES))
with Session(engine) as session:
    for head, core in tqdm(heads_cores, total=len(heads_cores), desc='writing database'):
        if ('GDGT' in core) or ('AR' in core):  # no chains
            pieces = dict(head=head, core=core)
            add_mol_as_comp(session, pieces)
        elif core in ('DGTS', 'BL', 'OL'):
            for chain1 in chain1s:
                pieces = dict(core=core, chain1=chain1)
                add_mol_as_comp(session, pieces)
        else:
            for chain1 in chain1s:
                for chain2 in chain2s:
                    pieces = dict(head=head, core=core, chain1=chain1, chain2=chain2)
                    add_mol_as_comp(session, pieces)
        session.commit()
