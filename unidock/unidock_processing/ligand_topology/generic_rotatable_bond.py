from rdkit import Chem


class GenericRotatableBond(object):
    def __init__(self):
        self.rotatable_bond_smarts = (
            "[!$(*#*)&!D1&!$([CH3])&!$(C(F)(F)F)&!$(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)&!$"
            "(C([CH3])([CH3])[CH3])]-!@[!$(*#*)&!D1&!$([CH3])&!$(C(F)(F)F)&!$"
            "(C(Cl)(Cl)Cl)&!$(C(Br)(Br)Br)&!$(C([CH3])([CH3])[CH3])]"
        )
        self.amide_bond_smarts = "[C&$(C=O)]-[N&$(NC=O);v3;H1,H2;+0]"
        self.amine_bond_smarts = "[*]-[N;$([+1;H3]),$([+0;H2])]"
        self.hydroxyl_bond_smarts = "[*]-[O,S;$([-1;H0]),$([+0;H1])]"
        self.carboxyl_bond_smarts = "[*]-[C;$(C(=[O,S])O)]"
        self.phosphate_bond_smarts = "[*]-[P;$(P(=[O,S])([O,S])[O,S])]"

        self.conjugate_aryl_amide_smarts_1 = (
            "[c,n;$(a:[c&H1,n&H0&D2])]-[C;$(C(=[O,S])N)]"
        )
        self.conjugate_aryl_amide_smarts_2 = (
            "[c,n;$(a:[c&H1,n&H0&D2])]-[N;$(NC(=[O,S]))]"
        )

        self.conjugate_bond_smarts = (
            "[C,N;0;$([C,N;0]=[C,N;0])]-[C,N;0;$([C,N;0]=[C,N;0])]"
        )

        self.rotatable_bond_pattern = Chem.MolFromSmarts(self.rotatable_bond_smarts)
        self.amide_bond_pattern = Chem.MolFromSmarts(self.amide_bond_smarts)
        self.amine_bond_pattern = Chem.MolFromSmarts(self.amine_bond_smarts)
        self.hydroxyl_bond_pattern = Chem.MolFromSmarts(self.hydroxyl_bond_smarts)
        self.carboxyl_bond_pattern = Chem.MolFromSmarts(self.carboxyl_bond_smarts)
        self.phosphate_bond_pattern = Chem.MolFromSmarts(self.phosphate_bond_smarts)

        self.conjugate_aryl_amide_pattern_1 = Chem.MolFromSmarts(
            self.conjugate_aryl_amide_smarts_1
        )
        self.conjugate_aryl_amide_pattern_2 = Chem.MolFromSmarts(
            self.conjugate_aryl_amide_smarts_2
        )

        self.conjugate_bond_pattern = Chem.MolFromSmarts(self.conjugate_bond_smarts)

    def identify_rotatable_bonds(self, mol, keep_h_rotors=False):
        """Rotatable bonds for the torsion tree.

        keep_h_rotors: keep terminal amine (-NH2/-NH3+) and hydroxyl/thiol
        (-OH/-SH) bonds rotatable. Only hydrogens move around them, so they are
        useless (and dilute the search) for Vina scoring, which ignores H; they
        are only worth keeping when a force-field minimization optimizes H
        positions. Default False matches upstream Uni-Dock2 and the
        Uni-Dock-Benchmarks inputs.
        """
        default_rotatable_bond_info_list = list(
            mol.GetSubstructMatches(self.rotatable_bond_pattern)
        )

        exclude_rotatable_bond_info_list = []
        exclude_rotatable_bond_info_list += list(
            mol.GetSubstructMatches(self.amide_bond_pattern)
        )
        if not keep_h_rotors:
            exclude_rotatable_bond_info_list += list(
                mol.GetSubstructMatches(self.amine_bond_pattern)
            )
            exclude_rotatable_bond_info_list += list(
                mol.GetSubstructMatches(self.hydroxyl_bond_pattern)
            )
        exclude_rotatable_bond_info_list += list(
            mol.GetSubstructMatches(self.phosphate_bond_pattern)
        )

        exclude_rotatable_bond_info_list += list(
            mol.GetSubstructMatches(self.conjugate_bond_pattern)
        )

        for exclude_rotatable_bond_info in exclude_rotatable_bond_info_list:
            exclude_rotatable_bond_info_reversed = tuple(
                reversed(exclude_rotatable_bond_info)
            )
            if exclude_rotatable_bond_info in default_rotatable_bond_info_list:
                default_rotatable_bond_info_list.remove(exclude_rotatable_bond_info)
            elif (
                exclude_rotatable_bond_info_reversed in default_rotatable_bond_info_list
            ):
                default_rotatable_bond_info_list.remove(
                    exclude_rotatable_bond_info_reversed
                )

        return default_rotatable_bond_info_list
