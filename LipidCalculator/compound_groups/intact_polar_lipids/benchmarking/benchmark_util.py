import os.path

from msIO import MSPReader, PeakList

from LipidCalculator.compound_groups.intact_polar_lipids.benchmarking.paths import path_folder_fragments


def pieces_from_ldg_name(name, use_second=False):
    """Convert names from LipidDatabase_Generator to LipidCalculator"""
    if use_second:
        if name.count(';') < 2:
            print(f'{name} has no secondary structure, using first')
            use_second = False
    # parse name
    name = name.split(';')[int(use_second) * 2].replace('-', ' ')

    if use_second:
        name = name.replace('(', ' ').replace('/', ' ').strip(')')

    # remove C- indicating core-something
    if name.startswith('C-'):
        name = name[2:]
    pieces = name.split()

    # add C for chains
    pieces = [f'C{n}' if n[0].isdigit() and ':' in n else n for n in pieces]
    return pieces


def get_fragment_from_lipiddatabase_generator(name: str) -> PeakList:
    head, core, *chains = name.split()
    # first, find the right file
    file_name = f'{head}-{core}_pos.msp'

    if len(chains) == 0:
        file_name = f'{head}-{core}_pos.msp'
    elif len(chains) == 1:
        name_msp = f'{head}-{core} {chains[0][1:]}'
    else:
        assert len(chains) == 2
        name_msp = f'{head}-{core}({chains[0][1:]}/{chains[1][1:]})'

    msp = MSPReader(os.path.join(path_folder_fragments, file_name), splitter_peaks_list=' ')

    mask_matches = msp.df_features.name.apply(lambda n: name_msp in n)
    assert mask_matches.sum() > 0, f'did not find {name} in {file_name}'
    idcs = msp.df_features.index[mask_matches]
    idx = idcs[0]
    ms_spec = msp.peak_list[idx]
    return ms_spec


if __name__ == '__main__':
    name = '2G DEG C26:0'
    spec = get_fragment_from_lipiddatabase_generator(name)
    ax = spec.plot()
    ax.set_title(name)
