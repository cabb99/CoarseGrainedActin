"""
System.py
Handles the system class for openMM
"""

# Global imports
import warnings

#Import openmm
try:
    import openmm
    import openmm.app
    from simtk import unit
except ModuleNotFoundError:
    import simtk.openmm as openmm
    import simtk.openmm.app
    import simtk.unit as unit

import pandas as pd
import numpy as np
import configparser
#import prody
import scipy.spatial.distance as sdist
import os
from . import utils
from typing import Optional



__author__ = 'Carlos Bueno'
__version__ = '0.3'
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
_ef = 1 * unit.kilocalorie / unit.kilojoule  # energy scaling factor
_df = 1 * unit.angstrom / unit.nanometer  # distance scaling factor
_af = 1 * unit.degree / unit.radian  # angle scaling factor


def canvas(with_attribution=True):
    """
    Placeholder function to show example docstring (NumPy format)

    Replace this function and doc string for your own project

    Parameters
    ----------
    with_attribution : bool, Optional, default: True
        Set whether or not to display who the quote is from

    Returns
    -------
    quote : str
        Compiled string including quote and optional attribution
    """

    quote = "The code is but a canvas to our imagination."
    if with_attribution:
        quote += "\n\t- Adapted from Henry David Thoreau"
    return quote

def create_actin(length: int=100,
                 twist: float=2.89942054,
                 shift: float=-28.21600347,
                 rotation: np.array=np.array([[1., 0., 0.],
                                    [0., 1., 0.],
                                    [0., 0., 1.]]),
                 translation: np.array=np.array([5000, 5000, 5000]),
                 template_file: str="coarseactin/data/CaMKII_bound_with_actin.csv",
                 abp: Optional[str]=None) -> pd.DataFrame:
    """
    Creates an actin fiber decorated (or not) with actin binding proteins.
    ----------
    length: integer, default: 100
        Number of actin monomers in the filament
    twist: float, default: 2.89942054,
        Helical angle between two succesive monomers in radians
    shift: float, default: -28.21600347,
        Helical distance between two succesive monomers in angstroms
    rotation: np.array, default: np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]),
        Rotation matrix (3x3) to rotate the actin filament. If unity the fiber will extend in the Z direction.
    translation: np.array, default:np.array([5000, 5000, 5000]),
        Translation matrix (3) to translate the actin filament after the rotation.
    template_file: str, defaul: "coarseactin/data/CaMKII_bound_with_actin.csv",
        File containing the information of a sample actin monomer decorated with ABPS
    abp: str, Optional, default: None
        Name of the bound actin binding protein

    Returns
    -------
    actin : pd.DataFrame
        DataFrame containing the coordinates of the actin filament
    """

    q = np.array([[np.cos(twist), -np.sin(twist), 0, 0],
                  [np.sin(twist), np.cos(twist), 0, 0],
                  [0, 0, 1, shift],
                  [0, 0, 0, 1]])
    rot = q[:3, :3].T
    trans = q[:3, 3]

    bound_actin_template = pd.read_csv(template_file, index_col=0)

    if abp is None:
        bound_actin_template = bound_actin_template[bound_actin_template['resName'].isin(['ACT'])]
    else:
        bound_actin_template = bound_actin_template[bound_actin_template['resName'].isin(['ACT', abp])]

    # Create the helix
    point = bound_actin_template[['x', 'y', 'z']]
    points = []
    for i in range(length):
        points += [point]
        point = np.dot(point, rot) + trans
    points = np.concatenate(points)

    # Create the model
    model = pd.DataFrame(points, columns=['x', 'y', 'z'])
    model["resSeq"] = [(j + i if name == 'ACT' else j) for i in range(length) for j, name in
                       zip(bound_actin_template["resSeq"], bound_actin_template["resName"])]
    model['chainID'] = [(0 if j == 'ACT' else i + 1) for i in range(length) for j in bound_actin_template["resName"]]
    model["name"] = [j for i in range(length) for j in bound_actin_template["name"]]
    model["type"] = [j for i in range(length) for j in bound_actin_template["type"]]
    model["resName"] = [j for i in range(length) for j in bound_actin_template["resName"]]
    model["element"] = [j for i in range(length) for j in bound_actin_template["element"]]

    # Remove two binding points and set last ends as ACD
    model = model[~(((model['resSeq'] >= length) | (model['resSeq'] <= 1)) & model['name'].isin(
        ['A5', 'A6', 'A7', 'Aa', 'Ab', 'Ac'])) &
                  ~(((model['chainID'] >= length) | (model['chainID'] == 1)) & ~model['resName'].isin(['ACT']))].copy()

    resmax = model[model['resName'].isin(['ACT'])]['resSeq'].max()
    resmin = model[model['resName'].isin(['ACT'])]['resSeq'].min()
    model.loc[model[(model['resSeq'] == resmax) & model['resName'].isin(['ACT'])].index, 'resName'] = 'ACD'
    model.loc[model[(model['resSeq'] == resmin) & model['resName'].isin(['ACT'])].index, 'resName'] = 'ACD'

    # Center the model
    model[['x', 'y', 'z']] -= model[['x', 'y', 'z']].mean()

    # Move the model
    model[['x', 'y', 'z']] = np.dot(model[['x', 'y', 'z']], rotation) + translation

    return model

def create_abp(rotation=np.array([[1., 0., 0.],
                                  [0., 1., 0.],
                                  [0., 0., 1.]]),
               translation=np.array([5000, 5000, 5000]),
               template_file="coarseactin/data/CaMKII_bound_with_actin.csv",
               abp='CaMKII'):
    """
    Creates an actin binding protein
    ----------
    rotation: np.array, default: np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]),
        Rotation matrix (3x3) to rotate the actin binding protein. If unity the fiber will extend in the Z direction.
    translation: np.array, default:np.array([5000, 5000, 5000]),
        Translation matrix (3) to translate the actin binding protein after the rotation.
    template_file: str, defaul: "coarseactin/data/CaMKII_bound_with_actin.csv",
        File containing the information of a sample actin monomer decorated with actin binding proteins.
    abp: str, default: 'CaMKII'
        Name of the actin binding protein

    Returns
    -------
    abp : pd.DataFrame
        DataFrame containing the coordinates of the actin fbinding protein
    """

    bound_actin_template = pd.read_csv(template_file, index_col=0)
    model = bound_actin_template[bound_actin_template['resName'].isin([abp])].copy()

    # Center the model
    model[['x', 'y', 'z']] -= model[['x', 'y', 'z']].mean()

    # Move the model
    model[['x', 'y', 'z']] = np.dot(model[['x', 'y', 'z']], rotation) + translation

    return model


if __name__ == "__main__":
    # Do something if this file is invoked on its own
    print(canvas())


class SystemData:
    """
    A class to store the system information, including atoms, coordinates and topology
    """

    def __init__(self, atoms, bonds=None, angles=None, dihedrals=None, impropers=None):
        self.atoms = atoms
        self.atoms.index = np.arange(1, len(self.atoms) + 1)
        self.masses = atoms[['type', 'mass']].drop_duplicates()
        self.masses.index = np.arange(1, len(self.masses) + 1)
        self.n_atoms = len(self.atoms)
        self.n_atomtypes = len(self.masses)

        if bonds is not None:
            self.bonds = bonds
            self.bonds.index = np.arange(1, len(self.bonds) + 1)
            self.bondtypes = bonds[['type', 'x0', 'k']].drop_duplicates()
            self.bondtypes.index = np.arange(1, len(self.bondtypes) + 1)
            self.n_bonds = len(self.bonds)
            self.n_bondtypes = len(self.bondtypes)
        else:
            self.bonds = pd.DataFrame()
            self.bondtypes = pd.DataFrame()
            self.n_bonds = 0
            self.n_bondtypes = 0

        if angles is not None:
            self.angles = angles
            self.angles.index = np.arange(1, len(self.angles) + 1)
            self.angletypes = angles[['type', 'x0', 'k']].drop_duplicates()
            self.angletypes.index = np.arange(1, len(self.angletypes) + 1)
            self.n_angles = len(self.angles)
            self.n_angletypes = len(self.angletypes)
        else:
            self.angles = pd.DataFrame()
            self.angletypes = pd.DataFrame()
            self.n_angles = 0
            self.n_angletypes = 0

        if dihedrals is not None:
            self.dihedrals = dihedrals
            self.dihedrals.index = np.arange(1, len(self.dihedrals) + 1)
            self.dihedraltypes = dihedrals[['type', 'x0', 'k']].drop_duplicates()
            self.dihedraltypes.index = np.arange(1, len(self.dihedraltypes) + 1)
            self.n_dihedrals = len(self.dihedrals)
            self.n_dihedraltypes = len(self.dihedraltypes)
        else:
            self.dihedrals = pd.DataFrame()
            self.dihedraltypes = pd.DataFrame()
            self.n_dihedrals = 0
            self.n_dihedraltypes = 0

        if impropers is not None:
            self.impropers = impropers
            self.impropers.index = np.arange(1, len(self.impropers) + 1)
            self.impropertypes = impropers[['type', 'x0', 'k']].drop_duplicates()
            self.impropertypes.index = np.arange(1, len(self.impropertypes) + 1)
            self.n_impropers = len(self.impropers)
            self.n_impropertypes = len(self.impropertypes)
        else:
            self.impropers = pd.DataFrame()
            self.impropertypes = pd.DataFrame()
            self.n_impropers = 0
            self.n_impropertypes = 0

        # self.n_bonds=len(self.bonds)
        # self.n_bondtypes=len(self.bondtypes)
        # self.xmin,self.xmax=atoms['x'].min(),atoms['x'].max()
        # self.ymin,self.ymax=atoms['y'].min(),atoms['y'].max()
        # self.zmin,self.zmax=atoms['z'].min(),atoms['z'].max()

    def write_data(self, file_name='actin.data', box_size=1000):
        self.xmin, self.xmax = 0, box_size
        self.ymin, self.ymax = 0, box_size
        self.zmin, self.zmax = 0, box_size
        with open(file_name, 'w+') as f:
            f.write('LAMMPS data file generated with python\n\n')
            f.write('\t%i atoms\n' % self.n_atoms)
            f.write('\t%i  bonds\n' % self.n_bonds)
            f.write('\t%i  angles\n' % self.n_angles)
            f.write('\t%i  dihedrals\n' % self.n_dihedrals)
            f.write('\t%i  impropers\n' % self.n_impropers)
            f.write('\n')
            f.write('\t%i  atom types\n' % self.n_atomtypes)
            f.write('\t%i  bond types\n' % self.n_bondtypes)
            f.write('\t%i  angle types\n' % self.n_angletypes)
            f.write('\t%i  dihedral types\n' % self.n_dihedraltypes)
            f.write('\t%i  improper types\n' % self.n_impropertypes)
            f.write('\n')
            f.write('\t %f %f xlo xhi\n' % (self.xmin, self.xmax))
            f.write('\t %f %f ylo yhi\n' % (self.ymin, self.ymax))
            f.write('\t %f %f zlo zhi\n' % (self.zmin, self.zmax))
            f.write('\n')
            f.write('Masses\n\n')
            for i, m in self.masses.iterrows():
                f.write('\t%i\t%f\n' % (i, m.mass))
            f.write('\n')
            f.write('Atoms\n\n')
            for i, a in self.atoms.iterrows():
                f.write('\t%i\t%i\t%i\t%f\t%f\t%f\t%f\n' % (i, a.molecule, a.type, a.q, a.x, a.y, a.z))
            f.write('\n')
            if self.n_bonds > 0:
                f.write('Bonds\n\n')
                for i, b in self.bonds.iterrows():
                    f.write('\t%i\t%i\t%i\t%i\n' % (i, b.type, b.i, b.j))
                f.write('\n')
            if self.n_angles > 0:
                f.write('Angles\n\n')
                for i, b in self.angles.iterrows():
                    f.write('\t%i\t%i\t%i\t%i\t%i\n' % (i, b.type, b.i, b.j, b.l))
                f.write('\n')
            if self.n_dihedrals > 0:
                f.write('Dihedrals\n\n')
                for i, b in self.dihedrals.iterrows():
                    f.write('\t%i\t%i\t%i\t%i\t%i\t%i\n' % (i, b.type, b.i, b.j, b.l, b.m))
                f.write('\n')
            if self.n_impropers > 0:
                f.write('Impropers\n\n')
                for i, b in self.impropers.iterrows():
                    f.write('\t%i\t%i\t%i\t%i\t%i\t%i\n' % (i, b.type, b.i, b.j, b.l, b.m))
                f.write('\n')

    def write_pdb(self, file_name='actin.pdb'):
        import string
        cc = (string.ascii_uppercase.replace('X', '') + string.ascii_lowercase + '1234567890' + 'X') * 1000
        cc_d = dict(zip(range(1, len(cc) + 1), cc))
        pdb_line = '%-6s%5i %-4s%1s%3s %1s%4i%1s   %8s%8s%8s%6.2f%6.2f          %2s%2s\n'
        pdb_atoms = self.atoms.copy()
        pdb_atoms['serial'] = np.arange(1, len(self.atoms) + 1)
        # pdb_atoms['name']       = self.atoms['type'].replace({1:'A1',2:'A2',3:'A3',4:'A4',5:'A5',6:'C1',7:'C2'})
        pdb_atoms['altLoc'] = ''
        # pdb_atoms['resName']    = self.atoms['molecule_name'].replace({'actin':'ACT','camkii':'CAM'})
        pdb_atoms['resName'] = self.atoms['resName']
        pdb_atoms['chainID'] = self.atoms['chainID']
        # pdb_atoms['resSeq']     = 0
        pdb_atoms['iCode'] = ''
        # pdb_atoms['x']          =
        # pdb_atoms['y']          =
        # pdb_atoms['z']          =
        pdb_atoms['occupancy'] = 0
        pdb_atoms['tempFactor'] = 0
        pdb_atoms['element'] = self.atoms['type'].replace(
            {1: 'C', 2: 'O', 3: 'N', 4: 'P', 5: 'H', 6: 'H', 7: 'H', 8: 'Mg', 9: 'Fe', 10: 'C'})
        pdb_atoms['charge'] = 0  # self.atoms['q'].astype(int)

        with open(file_name, 'w+') as f:
            chain = 'NoChain'
            resSeq = 0
            for i, a in pdb_atoms.iterrows():
                if a['chainID'] != chain:
                    resSeq = 1
                    chain = a['chainID']
                else:
                    resSeq += 1
                f.write(pdb_line % ('ATOM',
                                    int(a['serial']),
                                    a['name'].center(4),
                                    a['altLoc'],
                                    a['resName'],
                                    a['chainID'],
                                    a['resSeq'],
                                    a['iCode'],
                                    ('%8.3f' % (a['x'] / 10))[:8],
                                    ('%8.3f' % (a['y'] / 10))[:8],
                                    ('%8.3f' % (a['z'] / 10))[:8],
                                    a['occupancy'],
                                    a['tempFactor'],
                                    a['element'],
                                    a['charge']))

    def write_gro(self, file_name='actin.gro', box_size=1000):
        gro_line = "%5d%-5s%5s%5d%8s%8s%8s%8s%8s%8s\n"
        pdb_atoms = self.atoms.copy()
        pdb_atoms['resName'] = self.atoms[
            'resName']  # self.atoms['molecule_name'].replace({'actin':'ACT','camkii':'CAM'})
        # pdb_atoms['name']       = self.atoms['type'].replace({1:'Aa',2:'Ab',3:'Ca',4:'Cb',5:'Da',6:'Db'})
        pdb_atoms['serial'] = np.arange(1, len(self.atoms) + 1)
        pdb_atoms['chainID'] = self.atoms['molecule']
        self.xmin, self.xmax = 0, box_size
        self.ymin, self.ymax = 0, box_size
        self.zmin, self.zmax = 0, box_size
        resSeq = 0

        with open(file_name, 'w+') as f:
            f.write('Generated Model\n')
            f.write('%5i\n' % len(pdb_atoms))
            chain = 'NoChain'
            resSeq = 0
            for i, a in pdb_atoms.iterrows():
                if a['molecule'] != chain:
                    resSeq = 1
                    chain = a['molecule']
                else:
                    resSeq += 1
                f.write(gro_line % (a['molecule'],
                                    a['resName'],
                                    a['name'],
                                    int(a['serial']),
                                    ('%8.3f' % (a['x'] / 10))[:8],
                                    ('%8.3f' % (a['y'] / 10))[:8],
                                    ('%8.3f' % (a['z'] / 10))[:8],
                                    '', '', ''))
            f.write(('   ' + ' '.join(['%8.3f'] * 3) + '\n') % (self.xmax, self.ymax, self.zmax))

    def print_coeff(self):
        if self.n_bonds > 0:
            self.bondtypes = self.bondtypes.sort_values('type')
            for i, b in self.bondtypes.iterrows():
                print('bond_coeff', int(b.type), b.k, '%.4f' % b['x0'])
        if self.n_angles > 0:
            for i, b in self.angletypes.iterrows():
                print('angle_coeff', int(b.type), b.k, '%.4f' % b['x0'])
        if self.n_dihedrals > 0:
            for i, b in self.dihedraltypes.iterrows():
                print('dihedral_coeff', int(b.type), b.k, '%.4f' % b['x0'])
        if self.n_impropers > 0:
            for i, b in self.impropertypes.iterrows():
                print('improper_coeff', int(b.type), b.k, '%.4f' % b['x0'])


class CoarseActin:

    # @classmethod
    # def from_parameters(cls,
    #                     box_size=10000,
    #                     n_actins=10,
    #                     n_camkiis=200,
    #                     min_dist=200,
    #                     align_actins=False,
    #                     bundle=False,
    #                     system2D=False,
    #                     model='Binding-Qian2',
    #                     sname='actin',
    #                     actinLenMin=50,
    #                     actinLenMax=100):
    #     self = cls()
    #     # Get actin coordinates actin
    #     pdb = prody.parsePDB(f'{__location__}/3j8i.pdb')
    #     mean = np.array([])
    #     for chain in 'DEF':
    #         selection = pdb.select('chain %s' % chain)
    #         D1 = pdb.select('chain %s and (resid 1 to 32 or resid 70 to 144 or resid 338 to 375)' % chain)
    #         D2 = pdb.select('chain %s and (resid 33 to 69)' % chain)
    #         D3 = pdb.select('chain %s and (resid 145 to 180 or resid 270 to 337)' % chain)
    #         D4 = pdb.select('chain %s and (resid 181 to 269)' % chain)
    #         m1 = D1.getCoords().mean(axis=0)
    #         m2 = D2.getCoords().mean(axis=0)
    #         m3 = D3.getCoords().mean(axis=0)
    #         m4 = D4.getCoords().mean(axis=0)
    #         mean = np.concatenate([mean, m1, m2, m3, m4], axis=0)
    #     mean = mean.reshape(-1, 3)
    #     actin = pd.DataFrame(mean, columns=['x', 'y', 'z'])
    #     name = ['A1', 'A2', 'A3', 'A4'] * 3
    #     resid = [i for j in range(3) for i in [j] * 4]
    #     actin.index = zip(resid, name)
    #     # Build virtual sites
    #     vs = self.virtual_sites_definition
    #     for j in [2]:
    #         for i, s in vs[vs['molecule'] == model].iterrows():
    #             w12 = s['w12']
    #             w13 = s['w13']
    #             wcross = s['wcross']
    #
    #             a = actin.loc[[(j, s['p1'])]].squeeze() / 10
    #             b = actin.loc[[(j, s['p2'])]].squeeze() / 10
    #             c = actin.loc[[(j, s['p3'])]].squeeze() / 10
    #
    #             r12 = b - a
    #             r13 = c - a
    #             rcross = np.cross(r12, r13)
    #             r = (a + w12 * r12 + w13 * r13 + wcross * rcross) * 10
    #             r.name = (j, s['site'])
    #             actin = actin.append(r)
    #     actin_reference = actin.sort_index()
    #     # Build individual actins
    #     factin = []
    #     for i in range(n_actins):
    #         # Set actin length
    #         nactins = actinLenMin + int((actinLenMax - actinLenMin) * np.random.random())
    #
    #         names = ['A1', 'A2', 'A3', 'A4'] * 2 + ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7'] * (nactins - 2)
    #         resnames = ['ACD'] * (4 * 2) + ['ACT'] * (7 * (nactins - 2))
    #         resids = [1] * 4 + [2] * 4 + [i + 3 for j in range(nactins - 2) for i in [j] * 7]
    #         # actin_mass=41.74*1E3
    #         Factin = prody.AtomGroup()
    #         Factina = prody.AtomGroup()
    #         Factinb = prody.AtomGroup()
    #
    #         Factin.setCoords(actin_reference)
    #         Factina.setCoords(actin_reference[4:-3])
    #         Factinb.setCoords(actin_reference[:-4 - 3])
    #
    #         for i in range(nactins - 3):
    #             a, t = prody.superpose(Factina, Factinb)
    #             temp0 = Factin.getCoords()
    #             test = prody.applyTransformation(t, Factin)
    #             temp = np.concatenate([test.getCoords(), temp0[-4 - 3:]])
    #             # print(len(temp))
    #             Factin = prody.AtomGroup()
    #             Factina = prody.AtomGroup()
    #             Factinb = prody.AtomGroup()
    #             Factin.setCoords(temp)
    #             Factina.setCoords(temp[4:12])
    #             Factinb.setCoords(temp[0:8])
    #
    #         Factin = prody.AtomGroup()
    #         Factin.setCoords(temp[:])
    #         n = len(Factin)
    #         Factin.setNames(names)
    #         Factin.setResnames(resnames)
    #         Factin.setResnums(resids)
    #         # Factin.setNames(['ALA' for i in range(n)])
    #
    #         prody.writePDB('Factin.pdb', Factin)
    #         print(nactins, (n - 8) / 7. + 2)
    #
    #         atoms = pd.DataFrame(Factin.getCoords(), columns=['x', 'y', 'z'])
    #         atoms['q'] = -11
    #         atoms['molecule'] = 1
    #         atoms['type'] = [1, 2, 3, 4] * 2 + [1, 2, 3, 4, 5, 6, 7] * (nactins - 2)
    #         atoms['name'] = names
    #         # atoms['mass']=[D1_mass,D2_mass,D3_mass,D4_mass]*2+([D1_mass,D2_mass,D3_mass,D4_mass,0,0,0])*(nactins-2)
    #         atoms['resSeq'] = resids
    #         atoms['resName'] = resnames
    #         atoms.head()
    #         factin += [atoms.copy()]
    #
    #     # Read camkii
    #     camkii = self.template
    #     # Build box
    #     actins = []
    #     camkiis = []
    #     for i in range(n_actins):
    #         d = 0
    #         while d < min_dist:
    #             f = factin[i][['x', 'y', 'z']].copy()
    #             f = f - f.mean()
    #             if align_actins:
    #                 rot = utils.optimal_rotation(f)
    #             else:
    #                 rot = utils.random_rotation()
    #             f = pd.DataFrame(np.dot(rot, f[['x', 'y', 'z']].T).T, columns=f.columns)
    #             f = f - f.mean()
    #             f += [box_size / 2. for j in range(3)]
    #             a, b, c = [box_size * np.random.random() for j in range(3)]
    #             if bundle:
    #                 a = 0
    #             if system2D:
    #                 c = 0
    #             f += [a, b, c]
    #             f -= (f.mean() > box_size) * box_size
    #             f2 = factin[i].copy()
    #             f2[['x', 'y', 'z']] = f[['x', 'y', 'z']]
    #             # f+=[box_size/2. for i in range(3)]
    #             f2['molecule'] = i + 1
    #             f2['molecule_name'] = 'actin'
    #             f2['resName'] = factin[i]['resName']
    #             try:
    #                 d = sdist.cdist(f2[['x', 'y', 'z']], s[s['name'].isin(['A2', 'Cc'])][['x', 'y', 'z']]).min()
    #             except KeyError:
    #                 d = min_dist + 100
    #         actins += [f2]
    #         s = pd.concat(actins)
    #     print("Actins in system")
    #     print(f"Total number of particles: {len(s)}")
    #     for i in range(n_camkiis):
    #         d = 0
    #         while d < min_dist:
    #             f = camkii[['x', 'y', 'z']].copy()
    #             f = f - f.mean()
    #             f = pd.DataFrame(np.dot(random_rotation(), f[['x', 'y', 'z']].T).T, columns=f.columns)
    #             f = f - f.mean()
    #             f += [box_size / 2. for j in range(3)]
    #             a, b, c = [box_size * np.random.random() for j in range(3)]
    #             if system2D:
    #                 c = box_size / 10 * np.random.random()
    #             f += [a, b, c]
    #             f -= (f.mean() > box_size) * box_size
    #             f2 = camkii.copy()
    #             f2[['x', 'y', 'z']] = f[['x', 'y', 'z']]
    #             # f+=[box_size/2. for i in range(3)]
    #             f2['molecule'] = n_actins + i + 1
    #             f2['molecule_name'] = 'camkii'
    #             f2['resSeq'] = i + 1
    #             f2['resName'] = 'CAM'
    #             # f2['mass']/=100
    #             # rr=np.random.randint(2)
    #             # if rr==1:
    #             # f2['type']+=2
    #             d = sdist.cdist(f2[['x', 'y', 'z']], s[s['name'].isin(['A2', 'Cc'])][['x', 'y', 'z']]).min()
    #         camkiis += [f2]
    #         s = pd.concat(actins + camkiis, sort=True)
    #         print(f"CAMKII {i}")
    #     print("CAMKIIs in system")
    #     print(f"Total number of particles: {len(s)}")
    #     s.index = np.arange(1, len(s) + 1)
    #     s['mass'] = np.nan
    #     # Write system
    #     ss = SystemData(s.sort_values(['molecule', 'resSeq', 'name']))
    #     ss.write_data()
    #     ss.write_pdb(f'{sname}.pdb')
    #     ss.write_gro(f'{sname}.gro')
    #     ss.print_coeff()
    #     return self.from_topology(topology_file=f'{sname}.pdb', PlaneConstraint=system2D, periodic_box=box_size)

    @classmethod
    def from_topology(cls, topology_file='actin.pdb', periodic_box=None, PlaneConstraint=False):
        self = cls()
        if periodic_box is not None:
            self.periodic_box = [periodic_box * 0.1] * 3
        else:
            self.periodic_box = None
        self.forcefield = openmm.app.ForceField(f'{__location__}/data/ff.xml')
        if topology_file[-3:]=='pdb':
            self.top = openmm.app.PDBFile(topology_file)
        elif topology_file[-3:]=='cif':
            self.top = openmm.app.PDBxFile(topology_file)
        else:
            print('Unrecognized format for topology')
            raise IOError
        self.system = self.forcefield.createSystem(self.top.topology)
        if periodic_box is not None:
            self.system.setDefaultPeriodicBoxVectors(*np.diag(self.periodic_box))
        self.atom_list = self.parseTop()
        self.BuildVirtualSites()
        self.ComputeTopology()
        #self.setForces(PlaneConstraint=PlaneConstraint)
        return self
        # Parse topology data

    def parseConfigurationFile(self, configuration_file=f'{__location__}/data/actinff.conf'):
        """Reads the configuration file for the forcefield"""
        self.configuration_file = configuration_file
        config = configparser.ConfigParser()
        config.read(configuration_file)
        self.template = utils.parseConfigTable(config['Template'])
        self.bond_definition = utils.parseConfigTable(config['Bonds'])
        self.bond_definition['type'] = '1' + self.bond_definition['i'].astype(str) + '-' + \
                                       (1 + self.bond_definition['s']).astype(str) + \
                                       self.bond_definition['j'].astype(str)
        self.angle_definition = utils.parseConfigTable(config['Angles'])
        self.dihedral_definition = utils.parseConfigTable(config['Dihedrals'])
        self.repulsion_definition = utils.parseConfigTable(config['Repulsion'])
        self.virtual_sites_definition = utils.parseConfigTable(config['Virtual sites'])

        self.bond_definition = self.bond_definition[~self.bond_definition.molecule.isin(['Actin-ATP'])]
        self.angle_definition = self.angle_definition[~self.angle_definition.molecule.isin(['Actin-ATP'])]
        self.dihedral_definition = self.dihedral_definition[~self.dihedral_definition.molecule.isin(['Actin-ATP'])]
        # self.virtual_sites_definition = self.virtual_sites_definition[self.virtual_sites_definition.molecule.isin(['Actin-ADP', 'CaMKII','Binding-Qian2'])]

    def __init__(self):
        self.parseConfigurationFile()
        # self.forcefield = openmm.app.ForceField(f'{__location__}/ff.xml')
        # self.top = top
        # self.system = self.forcefield.createSystem(top.topology)
        # self.system.setDefaultPeriodicBoxVectors(*np.diag(periodic_box))
        # self.atom_list = self.parseTop()
        # self.BuildVirtualSites()
        # self.ComputeTopology()
        # self.setForces()
        # Parse topology data

    def parseTop(self):
        """ Converts the information from the topology to a table"""
        cols = ['atom_index', 'atom_id', 'atom_name',
                'residue_index', 'residue_id', 'residue_name',
                'chain_index', 'chain_id']
        data = []
        for residue in self.top.topology.residues():
            for atom in residue.atoms():
                data += [[atom.index, atom.id, atom.name,
                          residue.index, residue.id, residue.name,
                          residue.chain.index, residue.chain.id]]
        atom_list = pd.DataFrame(data, columns=cols)
        atom_list.index = atom_list['atom_index']
        return atom_list

    def BuildVirtualSites(self):
        """ Sets the parameters for the virtual sites"""
        virtual_sites_definition = self.virtual_sites_definition.copy()
        virtual_sites_definition.index = [tuple(b) for a, b in
                                          virtual_sites_definition[['molecule', 'site']].iterrows()]

        # Actin binding sites parameters

        w1 = np.array(virtual_sites_definition.loc[[('VothQianABP', 'A5')], ['w12', 'w13', 'wcross']].squeeze())
        w2 = np.array(virtual_sites_definition.loc[[('VothQianABP', 'A6')], ['w12', 'w13', 'wcross']].squeeze())
        w3 = np.array(virtual_sites_definition.loc[[('VothQianABP', 'A7')], ['w12', 'w13', 'wcross']].squeeze())
        w4 = np.array(virtual_sites_definition.loc[[('VothQianABP', 'Aa')], ['w12', 'w13', 'wcross']].squeeze())
        w5 = np.array(virtual_sites_definition.loc[[('VothQianABP', 'Ab')], ['w12', 'w13', 'wcross']].squeeze())
        w6 = np.array(virtual_sites_definition.loc[[('VothQianABP', 'Ac')], ['w12', 'w13', 'wcross']].squeeze())

        # CAMKII virtual sites
        cw1 = np.array(virtual_sites_definition.loc[[('CaMKII', 'C1')], ['w12', 'w13', 'wcross']].squeeze())
        cw2 = np.array(virtual_sites_definition.loc[[('CaMKII', 'C2')], ['w12', 'w13', 'wcross']].squeeze())
        cw3 = np.array(virtual_sites_definition.loc[[('CaMKII', 'C6')], ['w12', 'w13', 'wcross']].squeeze())
        cw4 = np.array(virtual_sites_definition.loc[[('CaMKII', 'C7')], ['w12', 'w13', 'wcross']].squeeze())

        # Virtual sites
        for chain_res, res in self.atom_list.groupby(['chain_index', 'residue_id']):
            assert len(res['residue_name'].unique()) == 1, print(len(res['residue_name'].unique()), _,
                                                                 res['residue_name'].unique())
            resname = res['residue_name'].unique()[0]
            ix = dict(list(zip(res['atom_name'], res['atom_index'])))
            if resname == 'ACT':
                # Virtual site positions
                a5 = openmm.OutOfPlaneSite(ix['A2'], ix['A1'], ix['A3'], w1[0], w1[1], w1[2])
                a6 = openmm.OutOfPlaneSite(ix['A2'], ix['A1'], ix['A3'], w2[0], w2[1], w2[2])
                a7 = openmm.OutOfPlaneSite(ix['A2'], ix['A1'], ix['A3'], w3[0], w3[1], w3[2])
                aa = openmm.OutOfPlaneSite(ix['A2'], ix['A1'], ix['A3'], w4[0], w4[1], w4[2])
                ab = openmm.OutOfPlaneSite(ix['A2'], ix['A1'], ix['A3'], w5[0], w5[1], w5[2])
                ac = openmm.OutOfPlaneSite(ix['A2'], ix['A1'], ix['A3'], w6[0], w6[1], w6[2])
                # Set up virtual sites
                self.system.setVirtualSite(ix['A5'], a5)
                self.system.setVirtualSite(ix['A6'], a6)
                self.system.setVirtualSite(ix['A7'], a7)
                self.system.setVirtualSite(ix['Aa'], aa)
                self.system.setVirtualSite(ix['Ab'], ab)
                self.system.setVirtualSite(ix['Ac'], ac)

            if resname == 'CAM' or (resname == 'CBP' and 'Ca' not in ix):
                # Parent sites
                c1 = ix['Cx1']
                c2 = ix['Cx2']
                c3 = ix['Cx3']
                # Virtual site positions
                c01 = openmm.OutOfPlaneSite(c1, c2, c3, cw1[0], cw1[1], cw1[2])
                c02 = openmm.OutOfPlaneSite(c1, c2, c3, cw2[0], cw2[1], cw2[2])
                c03 = openmm.OutOfPlaneSite(c2, c3, c1, cw1[0], cw1[1], cw1[2])
                c04 = openmm.OutOfPlaneSite(c2, c3, c1, cw2[0], cw2[1], cw2[2])
                c05 = openmm.OutOfPlaneSite(c3, c1, c2, cw1[0], cw1[1], cw1[2])
                c06 = openmm.OutOfPlaneSite(c3, c1, c2, cw2[0], cw2[1], cw2[2])
                c07 = openmm.OutOfPlaneSite(c1, c2, c3, cw3[0], cw3[1], cw3[2])
                c08 = openmm.OutOfPlaneSite(c1, c2, c3, cw4[0], cw4[1], cw4[2])
                c09 = openmm.OutOfPlaneSite(c2, c3, c1, cw3[0], cw3[1], cw3[2])
                c10 = openmm.OutOfPlaneSite(c2, c3, c1, cw4[0], cw4[1], cw4[2])
                c11 = openmm.OutOfPlaneSite(c3, c1, c2, cw3[0], cw3[1], cw3[2])
                c12 = openmm.OutOfPlaneSite(c3, c1, c2, cw4[0], cw4[1], cw4[2])
                cc = openmm.ThreeParticleAverageSite(c1, c2, c3, 1 / 3., 1 / 3., 1 / 3.)
                # Set up virtual positions
                self.system.setVirtualSite(ix['C01'], c01)
                self.system.setVirtualSite(ix['C02'], c02)
                self.system.setVirtualSite(ix['C03'], c03)
                self.system.setVirtualSite(ix['C04'], c04)
                self.system.setVirtualSite(ix['C05'], c05)
                self.system.setVirtualSite(ix['C06'], c06)
                self.system.setVirtualSite(ix['C07'], c07)
                self.system.setVirtualSite(ix['C08'], c08)
                self.system.setVirtualSite(ix['C09'], c09)
                self.system.setVirtualSite(ix['C10'], c10)
                self.system.setVirtualSite(ix['C11'], c11)
                self.system.setVirtualSite(ix['C12'], c12)
                self.system.setVirtualSite(ix['Cc'], cc)
            if resname == 'CAM2':
                # Parent sites
                c1 = ix['Cx1']
                c2 = ix['Cx2']
                c3 = ix['Cx3']
                # Virtual site positions
                c01 = openmm.OutOfPlaneSite(c1, c2, c3, cw1[0], cw1[1], cw1[2])
                c04 = openmm.OutOfPlaneSite(c2, c3, c1, cw2[0], cw2[1], cw2[2])
                c07 = openmm.OutOfPlaneSite(c1, c2, c3, cw3[0], cw3[1], cw3[2])
                c10 = openmm.OutOfPlaneSite(c2, c3, c1, cw4[0], cw4[1], cw4[2])
                cc = openmm.ThreeParticleAverageSite(c1, c2, c3, 1 / 3., 1 / 3., 1 / 3.)
                # Set up virtual positions
                self.system.setVirtualSite(ix['C01'], c01)
                self.system.setVirtualSite(ix['C04'], c04)
                self.system.setVirtualSite(ix['C07'], c07)
                self.system.setVirtualSite(ix['C10'], c10)
                self.system.setVirtualSite(ix['Cc'], cc)
            if resname == 'AAC':
                res2=self.atom_list[(self.atom_list['chain_index']==chain_res[0]) & (self.atom_list['residue_index']!=chain_res[1])]
                ix2 = dict(list(zip(res2['atom_name'], res2['atom_index'])))
                print(chain_res)
                print(res2)
                print(ix2)
                # Parent sites
                c1 = ix['Cb']
                c2 = ix2['Cb']
                print(ix)
                # Virtual site positions
                cy1 = openmm.TwoParticleAverageSite(c1, c2, 0.8, 0.2)
                cy2 = openmm.TwoParticleAverageSite(c1, c2, 0.6, 0.4)
                # Set up virtual positions
                self.system.setVirtualSite(ix['Cy1'], cy1)
                self.system.setVirtualSite(ix['Cy2'], cy2)
        self.atom_list['Virtual'] = [self.system.isVirtualSite(a) for a in range(len(self.atom_list))]

    def ComputeTopology(self):
        # print(bonds)
        # Bonds, angles and dihedrals
        bonds = []
        angles = []
        dihedrals = []
        for _, c in self.atom_list.groupby('chain_index'):
            ix = {}
            for name, aa in c.groupby('atom_name'):
                ix.update({name: list(aa.index)})

            for SB, B in zip([bonds, angles, dihedrals],
                             [self.bond_definition, self.angle_definition, self.dihedral_definition]):
                for _, b in B.iterrows():
                    temp = pd.DataFrame(columns=B.columns)
                    if 's' not in b:
                        b['s'] = 0

                    if b['i'] not in ix.keys():
                        continue

                    i1 = ix[b['i']][b['s']:]
                    i2 = ix[b['j']][:-b['s']] if b['s'] != 0 else ix[b['j']]
                    assert (len(i1) == len(i2))
                    temp['i'] = i1
                    temp['j'] = i2
                    if 'k' in b:
                        i3 = ix[b['k']]
                        assert (len(i1) == len(i3))
                        temp['k'] = i3
                    if 'l' in b:
                        i4 = ix[b['l']]
                        assert (len(i1) == len(i4))
                        temp['l'] = i4
                    for col in temp:
                        if col not in ['i', 'j', 'k', 'l']:
                            temp[col] = b[col]
                    SB += [temp]
        if len(bonds) > 0:
            bonds = pd.concat(bonds, sort=False)
            bonds.sort_values(['i', 'j'], inplace=True)
            bonds = bonds.reset_index(drop=True)
        else:
            bonds=pd.DataFrame()
        if len(angles)>0:
            angles = pd.concat(angles, sort=False)
            angles.sort_values(['i', 'j', 'k'], inplace=True)
            angles = angles.reset_index(drop=True)
        else:
            angles=pd.DataFrame()
        if len(dihedrals)>0:
            dihedrals = pd.concat(dihedrals, sort=False)
            dihedrals.sort_values(['i', 'j', 'k', 'l'], inplace=True)
            dihedrals = dihedrals.reset_index(drop=True)
        else:
            dihedrals=pd.DataFrame()
        self.bonds = bonds
        self.angles = angles
        self.dihedrals = dihedrals

    def Bond_diff(self, coord):
        # Comparison to starting structure distances
        import scipy.spatial.distance as sdist
        real_dist = sdist.squareform(sdist.pdist(coord.getPositions(asNumpy=True)))
        for i, b in self.bonds.iterrows():
            self.bonds.at[i, 'xr'] = real_dist[b['i'], b['j']] * 10
        self.bonds['diff'] = ((self.bonds['xr'] - self.bonds['r0']) ** 2) ** .5
        return self.bonds.groupby('type').mean().sort_values('diff', ascending=False)

    def clearForces(self):
        """ Removes all forces from the system """
        [self.system.removeForce(0) for i, f in enumerate(self.system.getForces())]

    def setForces(self, PlaneConstraint=False, forces=['multigaussian'], AlignmentConstraint=False):
        """ Adds the forces to the system """
        self.clearForces()
        # Harmonic Bonds
        harmonic_bond = openmm.HarmonicBondForce()
        harmonic_bond.setForceGroup(0)
        if self.periodic_box is not None:
            harmonic_bond.setUsesPeriodicBoundaryConditions(True)
        else:
            harmonic_bond.setUsesPeriodicBoundaryConditions(False)
        for i, b in self.bonds.iterrows():
            if b['potential']=='harmonic':
                harmonic_bond.addBond(int(b['i']), int(b['j']), b['r0'] / 10., b['K'] * 4.184 * 100)
        self.system.addForce(harmonic_bond)

        # Non-harmonic Bonds
        non_harmonic_bond = openmm.CustomBondForce("eps*(step(rmin-r)*(rmin-r)^2+step(r-rmax)*(r-rmax)^2)")
        non_harmonic_bond.setForceGroup(1)
        non_harmonic_bond.addPerBondParameter('rmin')
        non_harmonic_bond.addPerBondParameter('rmax')
        non_harmonic_bond.addPerBondParameter('eps')
        if self.periodic_box is not None:
            non_harmonic_bond.setUsesPeriodicBoundaryConditions(True)
        else:
            non_harmonic_bond.setUsesPeriodicBoundaryConditions(False)
        for i, b in self.bonds.iterrows():
            if b['potential'] == 'non_harmonic':
                non_harmonic_bond.addBond(int(b['i']), int(b['j']), [b['rmin'] / 10., b['rmax'] / 10., b['K'] * 4.184 * 100])
        self.system.addForce(non_harmonic_bond)

        # Harmonic angles
        harmonic_angle = openmm.HarmonicAngleForce()
        harmonic_angle.setForceGroup(2)
        if self.periodic_box is not None:
            harmonic_angle.setUsesPeriodicBoundaryConditions(True)
        else:
            harmonic_angle.setUsesPeriodicBoundaryConditions(False)
        for i, b in self.angles.iterrows():
            harmonic_angle.addAngle(int(b['i']), int(b['j']), int(b['k']), b['t0'] / 180 * np.pi, b['K'] * 4.184)
        self.system.addForce(harmonic_angle)

        # Harmonic torsions
        harmonic_torsion = openmm.PeriodicTorsionForce()
        harmonic_torsion.setForceGroup(3)
        if self.periodic_box is not None:
            harmonic_torsion.setUsesPeriodicBoundaryConditions(True)
        else:
            harmonic_torsion.setUsesPeriodicBoundaryConditions(False)
        for i, b in self.dihedrals.iterrows():
            harmonic_torsion.addTorsion(int(b['i']), int(b['j']), int(b['k']), int(b['l']), b['period'],
                                        b['t0'] / 180 * np.pi, b['K'] * 4.184)
        self.system.addForce(harmonic_torsion)

        # Repulsion
        for i, r in self.repulsion_definition.iterrows():
            # print(r)
            # print(r['force'].format(i))
            rf = openmm.CustomNonbondedForce('(epsilon{0}*((sigma{0}/r)^12-2*(sigma{0}/r)^6)+epsilon{0})*step(sigma{0}-r)'.format(i))
            rf.setForceGroup(4)
            if self.periodic_box is not None:
                rf.setNonbondedMethod(rf.CutoffPeriodic)
            else:
                rf.setNonbondedMethod(rf.CutoffNonPeriodic)
            rf.addGlobalParameter('epsilon{0}'.format(i), r['epsilon'])
            rf.addGlobalParameter('sigma{0}'.format(i), r['sigma'])
            rf.setCutoffDistance(10)
            rf.setUseLongRangeCorrection(False)
            for _, a in self.atom_list.iterrows():
                rf.addParticle()
            sel1 = self.atom_list[self.atom_list['atom_name'] == r['i']]
            sel2 = self.atom_list[self.atom_list['atom_name'] == r['j']]
            rf.addInteractionGroup(sel1.index, sel2.index)
            rf.createExclusionsFromBonds(self.bonds[['i', 'j']].values.tolist(), 3)
            self.system.addForce(rf)

        # Donors
        Cm = [self.atom_list[self.atom_list['atom_name'] == 'C01'].index,
              self.atom_list[self.atom_list['atom_name'] == 'C02'].index,
              self.atom_list[self.atom_list['atom_name'] == 'C03'].index,
              self.atom_list[self.atom_list['atom_name'] == 'C04'].index,
              self.atom_list[self.atom_list['atom_name'] == 'C05'].index,
              self.atom_list[self.atom_list['atom_name'] == 'C06'].index,
              self.atom_list[self.atom_list['atom_name'] == 'C07'].index,
              self.atom_list[self.atom_list['atom_name'] == 'C08'].index,
              self.atom_list[self.atom_list['atom_name'] == 'C09'].index,
              self.atom_list[self.atom_list['atom_name'] == 'C10'].index,
              self.atom_list[self.atom_list['atom_name'] == 'C11'].index,
              self.atom_list[self.atom_list['atom_name'] == 'C12'].index]
        Cc = self.atom_list[self.atom_list['atom_name'] == 'Cc'].index
        comb = [(0, 6), (1, 7), (2, 8), (3, 9), (4, 10), (5, 11)]

        if 'multigaussian' in forces: 
            print('multigaussian')

            for i, j in comb:
                gaussian = openmm.CustomHbondForce("-g_eps*g1;"
                                                         "g1=(exp(-dd/w1)+exp(-dd/w2))/2;"
                                                         "dd=(dist1^2+dist2^2+dist3^2)/3;"
                                                         "dist1= distance(a1,d1);"
                                                         "dist2= min(distance(a2,d2),distance(a2,d3));"
                                                         "dist3= min(distance(a3,d2),distance(a3,d3));")

                gaussian.setForceGroup(5)
                if self.periodic_box is not None:
                    gaussian.setNonbondedMethod(gaussian.CutoffPeriodic)
                else:
                    gaussian.setNonbondedMethod(gaussian.CutoffNonPeriodic)
                gaussian.addGlobalParameter('g_eps', 100)  # Energy minimum
                gaussian.addGlobalParameter('w1', 5.0)  # well1 width
                gaussian.addGlobalParameter('w2', 0.5)  # well2 width
                gaussian.setCutoffDistance(12)

                # Aceptors
                A1 = self.atom_list[self.atom_list['atom_name'] == 'A5'].index
                A2 = self.atom_list[self.atom_list['atom_name'] == 'A6'].index
                A3 = self.atom_list[self.atom_list['atom_name'] == 'A7'].index
                assert len(A1) == len(A2) == len(A3)
                for a1, a2, a3 in zip(A1, A2, A3):
                    gaussian.addAcceptor(a1, a2, a3)

                # Donors
                for d1, d2, d3 in zip(Cc, Cm[i], Cm[j]):
                    gaussian.addDonor(d1, d2, d3)

                self.system.addForce(gaussian)
        if 'doublegaussian' in forces:
            print('doublegaussian')

            for i, j in comb:
                gaussian = openmm.CustomHbondForce("-g_eps*g1;"
                                                         "g1=(exp(-dd/w1)+exp(-dd/w2))/2;"
                                                         "dd=(dist2^2+dist3^2)/2;"
                                                         "dist2= min(distance(a2,d2),distance(a2,d3));"
                                                         "dist3= min(distance(a3,d2),distance(a3,d3));")

                gaussian.setForceGroup(5)
                if self.periodic_box is not None:
                    gaussian.setNonbondedMethod(gaussian.CutoffPeriodic)
                else:
                    gaussian.setNonbondedMethod(gaussian.CutoffNonPeriodic)
                gaussian.addGlobalParameter('g_eps', 100)  # Energy minimum
                gaussian.addGlobalParameter('w1', 5.0)  # well1 width
                gaussian.addGlobalParameter('w2', 0.5)  # well2 width
                gaussian.setCutoffDistance(12)

                # Aceptors
                A1 = self.atom_list[self.atom_list['atom_name'] == 'A5'].index
                A2 = self.atom_list[self.atom_list['atom_name'] == 'A6'].index
                A3 = self.atom_list[self.atom_list['atom_name'] == 'A7'].index
                assert len(A1) == len(A2) == len(A3)
                for a1, a2, a3 in zip(A1, A2, A3):
                    gaussian.addAcceptor(a1, a2, a3)

                # Donors
                for d1, d2, d3 in zip(Cc, Cm[i], Cm[j]):
                    gaussian.addDonor(d1, d2, d3)

                self.system.addForce(gaussian)
        if 'singlegaussian' in forces:
            print('singleguassian')

            gaussian = openmm.CustomHbondForce("-g_eps*g1;"
                                                     "g1=(exp(-dd/w1)+exp(-dd/w2))/2;"
                                                     "dd= distance(a1,d1);")

            gaussian.setForceGroup(5)
            if self.periodic_box is not None:
                gaussian.setNonbondedMethod(gaussian.CutoffPeriodic)
            else:
                gaussian.setNonbondedMethod(gaussian.CutoffNonPeriodic)
            gaussian.addGlobalParameter('g_eps', 100)  # Energy minimum
            gaussian.addGlobalParameter('w1', 5.0)  # well1 width
            gaussian.addGlobalParameter('w2', 0.5)  # well2 width
            gaussian.setCutoffDistance(12)

            # Aceptors
            A1 = self.atom_list[self.atom_list['atom_name'] == 'A5'].index
            A2 = self.atom_list[self.atom_list['atom_name'] == 'A6'].index
            A3 = self.atom_list[self.atom_list['atom_name'] == 'A7'].index
            assert len(A1) == len(A2) == len(A3)
            for a1, a2, a3 in zip(A1, A2, A3):
                gaussian.addAcceptor(a1, -1, -1)

            # Donors
            for d1 in Cc:
                gaussian.addDonor(d1, -1, -1)

            self.system.addForce(gaussian)
        if 'abp' in forces: 
            print('ABP')
            gaussian = openmm.CustomHbondForce("-g_eps_ABP*g1;"
                                               "g1=(exp(-dd/w1)+exp(-dd/w2))/2;"
                                               "dd=(dist1^2+dist2^2+dist3^2)/3;"
                                               "dist1= distance(a1,d1);"
                                               "dist2= distance(a2,d2);"
                                               "dist3= distance(a3,d3);")

            gaussian.setForceGroup(5)
            if self.periodic_box is not None:
                gaussian.setNonbondedMethod(gaussian.CutoffPeriodic)
            else:
                gaussian.setNonbondedMethod(gaussian.CutoffNonPeriodic)
            gaussian.addGlobalParameter('g_eps_ABP', 100)  # Energy minimum
            gaussian.addGlobalParameter('w1', 5.0)  # well1 width
            gaussian.addGlobalParameter('w2', 0.5)  # well2 width
            gaussian.setCutoffDistance(12)

            # Aceptors
            A1 = self.atom_list[self.atom_list['atom_name'] == 'Aa'].index
            A2 = self.atom_list[self.atom_list['atom_name'] == 'Ab'].index
            A3 = self.atom_list[self.atom_list['atom_name'] == 'Ac'].index
            D1 = self.atom_list[self.atom_list['atom_name'] == 'Ca'].index
            D2 = self.atom_list[self.atom_list['atom_name'] == 'Cb'].index
            D3 = self.atom_list[self.atom_list['atom_name'] == 'Cd'].index

            assert len(A1) == len(A2) == len(A3)
            for a1, a2, a3 in zip(A1, A2, A3):
                gaussian.addAcceptor(a1, a2, a3)

            # Donors
            assert len(D1) == len(D2) == len(D3)
            for d1, d2, d3 in zip(D1, D2, D3):
                gaussian.addDonor(d1, d2, d3)

            self.system.addForce(gaussian)


        if PlaneConstraint:
            print(self.periodic_box)
            midz = self.periodic_box[-1] / 2 / 10
            print(midz)
            plane_constraint = openmm.CustomExternalForce('kp*(z-mid)^2')
            plane_constraint.setForceGroup(6)
            #if self.periodic_box is not None:
            #    plane_constraint.setUsesPeriodicBoundaryConditions(True)
            #else:
                #plane_constraint.setUsesPeriodicBoundaryConditions(False)
            plane_constraint.addGlobalParameter('mid', midz)
            plane_constraint.addGlobalParameter('kp', 0.001)
            for i in self.atom_list.index:
                plane_constraint.addParticle(i, [])
            self.system.addForce(plane_constraint)
        else:
            plane_constraint = openmm.CustomExternalForce('kp*0')
            plane_constraint.setForceGroup(6)
            #if self.periodic_box is not None:
            #    plane_constraint.setUsesPeriodicBoundaryConditions(True)
            #else:
            #    plane_constraint.setUsesPeriodicBoundaryConditions(False)
            plane_constraint.addGlobalParameter('kp', 0.001)
            for i in self.atom_list.index:
                plane_constraint.addParticle(i, [])
            self.system.addForce(plane_constraint)

        if AlignmentConstraint:
            print('Alignment Constraint added')
            alignment_constraint = openmm.CustomCentroidBondForce(2, 'kp_alignment*(distance(g1,g2)^2-(x1-x2)^2)')
            alignment_constraint.setForceGroup(7)
            alignment_constraint.addGlobalParameter('kp_alignment', 0.01)
            if self.periodic_box is not None:
                alignment_constraint.setUsesPeriodicBoundaryConditions(True)
            else:
                alignment_constraint.setUsesPeriodicBoundaryConditions(False)
            cc = 0
            for c, chain in self.atom_list.groupby('chain_index'):
                if 'ACT' in chain.residue_name.unique():
                    print(f'Setting up Alignment constraint for chain {c}')
                    alignment_constraint.addGroup(
                        list(chain[chain['atom_name'].isin(['A1', 'A2', 'A3', 'A4'])].index[:16]))
                    print(len(list(chain[chain['atom_name'].isin(['A1', 'A2', 'A3', 'A4'])].index[:16])))
                    alignment_constraint.addGroup(
                        list(chain[chain['atom_name'].isin(['A1', 'A2', 'A3', 'A4'])].index[-16:]))
                    print(len(list(chain[chain['atom_name'].isin(['A1', 'A2', 'A3', 'A4'])].index[-16:])))
                    print([cc, cc + 1])
                    alignment_constraint.addBond([cc, cc + 1])
                    cc += 2
            self.system.addForce(alignment_constraint)
            print(self.system.getNumForces())
        else:
            print('Alignment constrain not added')
            alignment_constraint = openmm.CustomCentroidBondForce(2, '0*kp_alignment*(distance(g1,g2)^2-(x1-x2)^2)')
            alignment_constraint.setForceGroup(7)
            alignment_constraint.addGlobalParameter('kp_alignment', 0.01)
            if self.periodic_box is not None:
                alignment_constraint.setUsesPeriodicBoundaryConditions(True)
            else:
                alignment_constraint.setUsesPeriodicBoundaryConditions(False)
            cc = 0
            for c, chain in self.atom_list.groupby('chain_index'):
                if 'ACT' in chain.residue_name.unique():
                    alignment_constraint.addGroup(
                        list(chain[chain['atom_name'].isin(['A1', 'A2', 'A3', 'A4'])].index[16:]))
                    alignment_constraint.addGroup(
                        list(chain[chain['atom_name'].isin(['A1', 'A2', 'A3', 'A4'])].index[-16:]))
                    alignment_constraint.addBond([cc, cc + 1])
                    cc += 2
            self.system.addForce(alignment_constraint)
            print(self.system.getNumForces())
        '''
	if AlignmentConstraint:
            alignment_constraint = openmm.CustomCompoundBondForce(2,'kb*((y1-y2)^2+(z1+z2)^2')
            alignment_constraint.addGlobalParameter('kb', 0.1)
            for i in self.atom_list.index:
                alignment_constraint.addBond(i, [])
            self.system.addForce(alignment_constraint)
        else:
            alignment_constraint = openmm.CustomExternalForce('kb*0')
            alignment_constraint.addGlobalParameter('kb', 0.1)
            for i in self.atom_list.index:
                alignment_constraint.addParticle(i, [])
            self.system.addForce(alignment_constraint)
        '''
