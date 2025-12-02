from msIO import MSPReader
from msIO.feature_managers.combined import ProjectImportManager

path_folder = r'\\hlabstorage.dmz.marum.de\scratch\Yannick\compounds\julius\fragments'

msp_manager = MSPReader()
project_import_manager = ProjectImportManager(mgf_manager=msp_manager)

initiate_db(db_file)
project_import_manager.to_sql(db_file, feature_ids=project_import_manager.feature_ids[:10])
# project_import_manager.to_sql(db_file)


f = project_import_manager.active_managers['metaboscape'].get_feature(1)
