from LipidCalculator.GuaymasSurface.known_compounds import SELECTED_COMPOUND_NAMES_REVERSED
from LipidCalculator.GuaymasSurface.paths import db_file_surface
from LipidCalculator.rdkit.cleaving.generate_fragments import plot_ms2_prediction, predict_ms2
from LipidCalculator.compound_groups.intact_polar_lipids.benchmarking.benchmark_util import \
    get_fragment_from_lipiddatabase_generator
from LipidCalculator.compound_groups.intact_polar_lipids.generate_ipl import ipl_automatic_bonds
from msIO import PeakList
from msIO.features.metaboscape import FeatureMetaboScape
from msIO.features.mgf import FeatureMgf, MsSpec
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, joinedload

# annotated name to in silico name
# annotations can be the same for multiple features


print('fetching features with annotations...')
# fetch synthetic spectra for annotated ones
# get features with annotation
engine_surface = create_engine('sqlite:///' + db_file_surface, echo=False)
with Session(engine_surface) as session:
    feature_ids_and_names = session.query(FeatureMetaboScape.feature_id, FeatureMetaboScape.name_metaboscape).filter(
        FeatureMetaboScape.name_metaboscape.isnot(None)).all()
feature_id_to_names = dict(feature_ids_and_names)
feature_id_to_names = {k: v for k, v in feature_id_to_names.items() if '?' not in v}

print('fetching MS2 spectra for features with annotations...')
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


# %%
def create_plot_for(name_syn):
    # fetch metabo name
    name_metabo = SELECTED_COMPOUND_NAMES_REVERSED[name_syn]

    f_ids = [f_id
             for f_id, _name in feature_id_to_names.items()
             if _name == name_metabo]
    ms_specs_measured = [feature_id_to_ms2[f_id] for f_id in f_ids]

    mol = ipl_automatic_bonds(name_syn.split(), split_chain=True)
    # TODO: use correct adduct
    fig, ax = plt.subplots(figsize=(12, 8))
    ms = predict_ms2(
        mol=mol,
        adduct_type='[M+H]+',
        max_recursion_depth=1,
        allow_charge_relocation=True,
        cleavage_types=['INDUCTIVE', 'ALPHA']
    )
    plot_ms2_prediction(ms, add_struct_plots=False, ax=ax)
    # add the prediction from Julius' DB
    try:
        frag_julius: PeakList = get_fragment_from_lipiddatabase_generator(name_syn)
        ints = [i / 1000 for i in frag_julius.intensities]
        ax.stem(frag_julius.mzs, ints, linefmt='k-.', markerfmt='')
        # add annotations
        for mz, i, txt in zip(frag_julius.mzs, ints, frag_julius.annotations):
            ax.text(mz, .8 * i, txt, rotation=45)
    except FileNotFoundError as e:
        print(f'could not add other synthetic spectrum: {e}')
    except ValueError as e:
        print(f'could not add other synthetic spectrum: {e}')

    M = ExactMolWt(mol)
    mz_h_plus = (M + 1.007825 - .0005)
    for i, ms_spec_measured in enumerate(ms_specs_measured):
        iis = np.array(ms_spec_measured.peaks.intensities)
        ax.stem(
            ms_spec_measured.peaks.mzs,
            -iis / iis.max(),
            markerfmt='none',
            linefmt=f'C{i}',
            label=r'$\Delta m/z =$' + f'{ms_spec_measured.mz - mz_h_plus:.4f}'
        )
    ax.set_title(f'{name_syn} (M = {M:.4f})')
    ax.legend()
    return fig, ax


# %% plot measured and synthetic
import matplotlib.pyplot as plt
import numpy as np
from rdkit.Chem.Descriptors import ExactMolWt

create_plot_for(name_syn='1G DAG C17:1 C16:0')
plt.show()

# for name_metabo in tqdm(feature_id_to_names.values(), desc='creating plots'):
#     if name_metabo not in SELECTED_COMPOUND_NAMES:
#         continue
#     name_syn = SELECTED_COMPOUND_NAMES[name_metabo]
#     if 'AR' in name_syn:
#         continue
#     fig, ax = create_plot_for_name_metabo(name_metabo, name_syn)
#     plt.savefig(os.path.join(folder, f'{name_syn.replace(':', 'dd')}.png'), dpi=600)
#     plt.close()
