"""The structure behind a lipid name: building blocks that ipl_automatic_bonds turns into a molecule."""
from dataclasses import dataclass, field

from rdkit.Chem import Mol


class NotSupported(ValueError):
    """The name cannot be mapped to building blocks"""


@dataclass(frozen=True)
class Lipid:
    """Head groups (up to two, the second one is bonded to the second attachment point of the core, e.g. the other
    glycerol of tetraethers), core and chains like C16:0 (a single chain is a sum composition and is split between
    the attachment points), core_chains sets the isoprenoid chains of the core (e.g. '20:0_25:0' for archaeol),
    core_double_bonds the double bonds in the biphytanyl chains of tetraethers, core_hydroxy the hydroxyl groups at C3
    of the isoprenoid chains (OH-GDGTs, hydroxyarchaeols), core_methyls the additional methyl groups at C13 of the
    biphytanyl chains (Homo-Me-GDGTs)."""
    heads: tuple[str, ...] = ()
    core: str = ''
    chains: tuple[str, ...] = ()
    core_chains: str | None = None
    core_double_bonds: int = 0
    core_hydroxy: int = 0
    core_methyls: int = 0
    # ArchLips isomer label (e.g. 'a' of Gly-2OH-GDD-0a) if the building blocks do not define the isomer (the core
    # piece is a representative structure, e.g. the head group may be at another OH)
    isomer: str | None = None
    # lipid class of the name it was parsed from (e.g. 'archaeol', 'tetraether', 'MS-DIAL PC'), not part of the identity
    family: str = field(default='', compare=False)

    def pieces(self) -> list[str]:
        return [*self.heads, self.core, *self.chains]

    def build_kwargs(self) -> dict:
        """Keyword arguments for ipl_automatic_bonds"""
        kwargs = dict(split_chain=len(self.chains) == 1, sort_chains=False)
        if self.core_chains:
            kwargs['core_chains'] = self.core_chains
        if self.core_double_bonds:
            kwargs['core_double_bonds'] = self.core_double_bonds
        if self.core_hydroxy:
            kwargs['core_hydroxy'] = self.core_hydroxy
        if self.core_methyls:
            kwargs['core_methyls'] = self.core_methyls
        return kwargs

    def mol(self) -> Mol:
        from LipidCalculator.compound_groups.intact_polar_lipids.generate_ipl import ipl_automatic_bonds
        return ipl_automatic_bonds(self.pieces(), **self.build_kwargs())

    def core_only(self) -> 'Lipid':
        """The core lipid (without head groups)"""
        return Lipid(core=self.core, chains=self.chains, core_chains=self.core_chains,
                     core_double_bonds=self.core_double_bonds, core_hydroxy=self.core_hydroxy,
                     core_methyls=self.core_methyls, family=self.family)
