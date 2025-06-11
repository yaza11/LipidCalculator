class AdductSeries:
    def __init__(self):
        pass

    def get_adducts(
            self,
            adducts: list[str] | None = None,
            ionization: str = '+'
    ) -> dict[str, float]:
        if adducts is None:
            adducts = ['H', 'K', 'Na', 'NH4']
        formulas = [adduct + ionization for adduct in adducts]
        masses = [self.mass + self.get_mass(formula) for formula in formulas]
        return dict(zip(adducts, masses))
