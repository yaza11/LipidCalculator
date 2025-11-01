"""Use msIOs Compound database infrastructure to create an SQL file"""
from msIO.annotations.main import SqlBaseClassComp, Compound, IonPeak, IsotopePeak, FragmentPeak, CompoundGroup
from sqlalchemy import create_engine
import os

from sqlalchemy.orm import Session


def submit_to_db_inside_session(session, compound: Compound, ions: list[IonPeak],
                                frags: dict[IonPeak, list[FragmentPeak]],
                                compound_groups: list[CompoundGroup]):
    for compound_group in compound_groups:
        compound.compound_groups.append(compound_group)
    for ion in ions:
        # connect ions to compound
        compound.ions.append(ion)
        # right now we are not really generating isotope peaks, so we can just create one artificially here
        iso_peak = IsotopePeak(isotope_order=0, mz=ion.mz)
        ion.isotopes.append(iso_peak)

        _frags = frags[ion]
        for frag in _frags:
            iso_peak.fragments.append(frag)
        session.add(compound)


def submit_to_db(engine, compound: Compound, ions: list[IonPeak], frags: dict[IonPeak, list[FragmentPeak]],
                 compound_groups: list[CompoundGroup]):
    with Session(engine) as session:
        submit_to_db_inside_session(session, compound, ions, frags, compound_groups)
        session.commit()


def get_engine(db_path_file):
    engine = create_engine(f"sqlite+pysqlite:///{db_path_file}", echo=False)
    return engine


if __name__ == '__main__':
    path_folder = r'\\hlabstorage.dmz.marum.de\scratch\Yannick\compounds\LipidCalculator'
    db_path_file = os.path.join(path_folder, 'compound_database.db')

    engine = create_engine(f"sqlite+pysqlite:///{db_path_file}", echo=False)

    SqlBaseClassComp.metadata.create_all(engine)
