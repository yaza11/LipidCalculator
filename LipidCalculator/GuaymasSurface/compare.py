from LipidCalculator.cleaving.generate_fragments import predict_ms2, plot_ms2_prediction
from LipidCalculator.compound_groups.intact_polar_lipids.benchmarking.benchmark_util import \
    get_fragment_from_lipiddatabase_generator
from LipidCalculator.compound_groups.intact_polar_lipids.generate_ipl import ipl_automatic_bonds
from msIO import MgfImportManager, PeakList
from msIO.annotations.main import FragmentPeak, IonPeak, Compound, Molecule, IsotopePeak, CompoundGroup
from msIO.feature_managers.combined import ProjectImportManager
from msIO.feature_managers.metaboscape import MetaboscapeImportManager
from msIO.features.metaboscape import FeatureMetaboScape
from msIO.features.mgf import FeatureMgf, MsSpec
from msIO.sql.session import initiate_db
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, selectinload, joinedload
from tqdm import tqdm

# path_file_in_silico_db = r"C:\Users\yanni\Downloads\database_test.db"

path_metaboscape_csv_surface = r'\\hlabstorage.dmz.marum.de\scratch\Yannick\Janina Guaymas\surface_export\Guaymas Basin 2024_Neumessung ShortRP.csv'
path_mgf_sirius_surface = r"\\hlabstorage.dmz.marum.de\scratch\Yannick\Janina Guaymas\surface_export\Guaymas Basin 2024_Neumessung ShortRP.sirius.mgf"
db_file_surface = r'\\hlabstorage.dmz.marum.de\scratch\Yannick\Janina Guaymas\sql\database.sqlite'

# annotated name to in silico name
# annotations can be the same for multiple features
SELECTED_COMPOUND_NAMES = {
    'PI-AEG C34:2': 'PI AEG C34:2',
    'PC16:1_16:1': 'PC DAG C16:1 C16:1',
    'PG-DEG C32:1': 'PG DEG C32:1',
    'PC 16:0_16:1': 'PC DAG C16:0 C16:1',
    'PI-AEG C36:2': 'PI AEG C36:2',
    'PI-DEG C34:2': 'PI DEG C34:2',
    'PI-AR': 'PI AR',
    'PE-AR [M+H]+': 'PE AR',
    'PG-DAG 32:2; [M+NH4]+; PG-DAG(16:1/16:1)': 'PG DAG C16:1 C16:1',
    '1G-DAG 33:2; [M+H]+; 1G-DAG(17:1/16:1)': '1G DAG C17:1 C16:1',
    'PG 16:1_18:2': 'PG DAG C16:1 C18:2',
    'PE-DAG 30:2; [M+H]+; PE-DAG(16:1/14:1)': 'PE DAG C16:1 C14:1',
    '1G-DAG 33:0; [M+H]+; 1G-DAG(17:0/16:0)': '1G DAG C17:0 C16:0',
    'PG-DAG32:1;[M+NH4]+;PG-DAG(16:1/16:0)': 'PG DAG C16:1 C16:0',
    '1G-DAG 33:1; [M+H]+; 1G-DAG(17:1/16:0)': '1G DAG C17:1 C16:0',
    'PE-DAG 30:1; [M+H]+; PE-DAG(16:1/14:0)': 'PE DAG C16:1 C14:0',
    'PG-DAG 34:1; [M+H]+; PG-DAG(26:1/8:0)': 'PG DAG C26:1 C8:0',
    'PI-AEG C32:1': 'PI AEG C32:1',
    'PG-AEG-C33:2': 'PG AEG C33:2',
    'PE-DAG32:2;[M+H]+;PE-DAG(16:1/16:1)': 'PE DAG C16:1 C16:1',
    'PE-DAG 31:1; [M+H]+; PE-DAG(16:1/15:0)': 'PE DAG C16:1 C15:0',
    'PE 16:1_17:1': 'PE DAG C16:1 C17:1',
    'PC16:0_20:5': 'PC DAG C16:0 C20:5',
    'PG 18:1/18:1': 'PG DAG C18:1 C18:1',
    'PC 18:1_20:5': 'PC DAG C18:1 C20:5',
    'PE-DAG 33:2; [M+H]+; PE-DAG(17:1/16:1)': 'PE DAG C17:1 C16:1',
    'PE-DAG32:1;[M+H]+;PE-DAG(16:1/16:0)': 'PE DAG C16:1 C16:0',
    'PME-DAG 32:1; [M+H]+; PME-DAG(16:1/16:0)': 'PME DAG C16:1 C16:0',
    'PE-DAG34:2;[M+H]+;PE-DAG(18:1/16:1)': 'PE DAG C18:1 C16:1',
    'PDME-DAG34:2;[M+H]+;PDME-DAG(16:1/18:1)': 'PDME DAG C16:1 C18:1',
    'PME-DAG 34:2; [M+H]+; PME-DAG(18:1/16:1)': 'PME DAG C18:1 C16:1',
    'PC 17:1_17:1': 'PC DAG C17:1 C17:1',
    'PC 16:0_17:1': 'PC DAG C16:0 C17:1',
    'PC16:0_16:0': 'PC DAG C16:0 C16:0',
    'PI-OH-AR': 'PI OH-AR',
    'PI-AEG-C35:0': 'PI AEG C35:0',
    'PC 16:0_18:1': 'PC DAG C16:1 C18:1',
    'PG-OH-AR': 'PG OH-AR',
    'PS-OH-AR': 'PS OH-AR',
    'PI-DEG C35:1': 'PI DEG C35:1',
    '2OH-AR Na+': '2OH-AR',
    '2G-OH-AR': '2G OH-AR',
    '1G-OH-AR [M+NH4]+': '1G OH-AR',
    'PG-AR': 'PG AR',
    'C-OH-AR': 'OH-AR',
    '2G-AR [M+NH4]+': '2G AR',
    '1G-AR [M+NH4]+': '1G AR',
    'PC 21:0_21:0': 'PC DAG C21:0 C21:0',
}


def create_surface_sql():
    # write sql database for surface samples
    metaboscape = MetaboscapeImportManager(path_metaboscape_csv_surface)
    mgf = MgfImportManager(path_mgf_sirius_surface)

    project_import_manager = ProjectImportManager(mgf_manager=mgf,
                                                  metaboscape_manager=metaboscape)

    initiate_db(db_file_surface)
    project_import_manager.to_sql(db_file_surface)


print('fetching features with names...')
# fetch synthetic spectra for annotated ones
# get features with annotation
engine_surface = create_engine('sqlite:///' + db_file_surface, echo=False)
with Session(engine_surface) as session:
    feature_ids_and_names = session.query(FeatureMetaboScape.feature_id, FeatureMetaboScape.name_metaboscape).filter(
        FeatureMetaboScape.name_metaboscape.isnot(None)).all()
feature_id_to_names = dict(feature_ids_and_names)
feature_id_to_names = {k: v for k, v in feature_id_to_names.items() if '?' not in v}

print('fetching MS2 spectra for features with names...')
query = select(FeatureMgf.feature_id, MsSpec).join(
    FeatureMgf,
    FeatureMgf.id == MsSpec.feature_mgf_id,
    full=True
).filter(
    FeatureMgf.feature_id.in_(list(feature_id_to_names.keys()))
).filter(
    MsSpec.ms_level == 2  # filter for MS2 spectra
).options(
    joinedload(MsSpec.peaks).subqueryload(PeakList.peaks)
)

# fetch corresponding ms2 spectra
with Session(engine_surface) as session:
    ms_specs = session.execute(query).all()
feature_id_to_ms2 = dict(ms_specs)
feature_id_to_ion_mz = {k: v.mz for k, v in feature_id_to_ms2.items()}
# some features have Metaboscape object but not Mgf
feature_id_to_names = {k: v for k, v in feature_id_to_names.items() if k in feature_id_to_ion_mz}

# get adducts
with Session(engine_surface) as session:
    # TODO:
    ...

# %% plot measured and synthetic
import matplotlib.pyplot as plt
import numpy as np
import os
from rdkit.Chem.Descriptors import ExactMolWt

folder = r'C:\Users\Yannick Zander\Downloads\figures'
for name_metabo in tqdm(feature_id_to_names.values(), desc='creating plots'):
    if name_metabo not in SELECTED_COMPOUND_NAMES:
        continue
    f_ids = [f_id
             for f_id, _name in feature_id_to_names.items()
             if _name == name_metabo]
    ms_specs_measured = [feature_id_to_ms2[f_id] for f_id in f_ids]

    name_syn = SELECTED_COMPOUND_NAMES[name_metabo]
    mol = ipl_automatic_bonds(name_syn.split(), split_chain=True)
    # TODO: use correct adduct
    ms = predict_ms2(mol=mol, adduct_type='[M+H]+', max_recursion_depth=1)

    fig, ax = plt.subplots(figsize=(12, 8))
    plot_ms2_prediction(ms, add_struct_plots=False, ax=ax)
    # add the prediction from Julius' DB
    try:
        frag_julius: PeakList = get_fragment_from_lipiddatabase_generator(name_syn)
        ints = [i / 1000 for i in frag_julius.intensities]
        ax.stem(frag_julius.mzs, ints, linefmt='k', markerfmt='')
        # add annotations
        for mz, i, txt in zip(frag_julius.mzs, ints, frag_julius.annotations):
            ax.text(mz, .8 * i, txt, rotation=45)
    except FileNotFoundError as e:
        print(f'could not add other synthetic spectrum: {e}')

    for i, ms_spec_measured in enumerate(ms_specs_measured):
        iis = np.array(ms_spec_measured.peaks.intensities)
        ax.stem(
            ms_spec_measured.peaks.mzs,
            -iis/iis.max(),
            markerfmt='none',
            linefmt=f'C{i}'
        )
    M = ExactMolWt(mol)
    ax.set_title(f'{name_syn} (M = {M:.4f}, Delta m/z = {M + 1.007825 - .0005 - ms_spec_measured.mz:.4f})')
    plt.savefig(os.path.join(folder, f'{name_syn.replace(':', 'dd')}.png'), dpi=600)
    plt.close()
