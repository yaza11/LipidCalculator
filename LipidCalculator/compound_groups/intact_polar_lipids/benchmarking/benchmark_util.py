def pieces_from_ldg_name(name, use_second=False):
    if use_second:
        if name.count(';') < 2:
            print(f'{name} has no secondary structure')
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
