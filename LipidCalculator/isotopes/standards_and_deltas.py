f_VPDB_C13: float = 0.0111802
f_VSMOV_O17: float = 379.9e-6  # https://en.wikipedia.org/wiki/Vienna_Standard_Mean_Ocean_Water
f_VSMOV_H2: float = 155.76e-6  # https://en.wikipedia.org/wiki/Vienna_Standard_Mean_Ocean_Water
f_V_CDT: float = 1 / 22.220  # not sure if this is the correct value https://en.wikipedia.org/wiki/%CE%9434S
ATOM2ISOS: dict[str, tuple[str, str]] = {
    'H': ('H[1]', 'H[2]'),
    'C': ('C[12]', 'C[13]'),
    'O': ('O[16]', 'O[17]'),
    'S': ('S[32]', 'S[34]')
}
DEFAULT_MASS_TOLERANCE: float = 5e-3


def C13C12to_delta13C(pC12, pC13):
    """Calculate delta value from C13 and C12 portions"""
    return (pC13 / pC12 / f_VPDB_C13 - 1) * 1000  # =delta13C in permil


def H2H1to_delta13C(pH1, pH2):
    """Calculate delta value """
    return (pH2 / pH1 / f_VSMOV_H2 - 1) * 1000


def O17O16to_delta17O(pO16, pO17):
    """Calculate delta value """
    return (pO17 / pO16 / f_VSMOV_O17 - 1) * 1000


def S34S32to_delta34S(pS32, pS34):
    """Calculate delta value """
    return (pS34 / pS32 / f_V_CDT - 1) * 1000


def delta13C_to_f(delta13C):
    """Calculate fraction of C13 and C12 from delta value"""
    f = (delta13C / 1000 + 1) * f_VPDB_C13
    # since C13 + C12 = 1
    C13 = f / (1 + f)
    C12 = 1 - C13
    return C13, C12


def delta2H_to_f(delta2H):
    """Calculate fraction of C13 and C12 from delta value"""
    f = (delta2H / 1000 + 1) * f_VSMOV_H2
    # since 2H + H = 1
    H2 = f / (1 + f)
    H1 = 1 - H2
    return H2, H1


def delta17O_to_f(delta17O):
    """Calculate fraction of C13 and C12 from delta value"""
    f = (delta17O / 1000 + 1) * f_VSMOV_O17
    # since 2H + H = 1
    O17 = f / (1 + f)
    O16 = 1 - O17
    return O17, O16


def delta34S_to_f(delta34S):
    f = (delta34S / 1000 + 1) * f_V_CDT
    S34 = f / (1 + f)
    S32 = 1 - S34
    return S34, S32
