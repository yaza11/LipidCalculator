from msIO import MgfImportManager
from msIO.feature_managers.combined import ProjectImportManager
from msIO.feature_managers.metaboscape import MetaboscapeImportManager
from msIO.sql.session import initiate_db

from LipidCalculator.GuaymasSurface.paths import path_metaboscape_csv_surface, path_mgf_sirius_surface, db_file_surface


def create_surface_sql():
    # write sql database for surface samples
    metaboscape = MetaboscapeImportManager(path_metaboscape_csv_surface)
    mgf = MgfImportManager(path_mgf_sirius_surface)

    project_import_manager = ProjectImportManager(mgf_manager=mgf,
                                                  metaboscape_manager=metaboscape)

    initiate_db(db_file_surface)
    project_import_manager.to_sql(db_file_surface)


if __name__ == '__main__':
    create_surface_sql()
