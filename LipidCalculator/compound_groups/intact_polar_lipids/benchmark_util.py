def pieces_from_ldg_name(name):
    # remove C- indicating core-something
    if name.startswith('C-'):
        name = name[2:]

    # parse name
    pieces = name.split(';')[0].replace('-', ' ').split()
    # add C for chains
    pieces = [f'C{n}' if n[0].isdigit() and ':' in n else n for n in pieces]
    return pieces
