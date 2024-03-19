import pandas
import numpy as np
import io

"""
Python library to allow easy handling of coordinate files for molecular dynamics using pandas DataFrames.
"""
if __name__ == "__main__":
    import utils
else:
    from . import utils

_protein_residues = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
                     'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
                     'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
                     'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
                     'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}


class Scene(pandas.DataFrame):
    # Initialization
    def __init__(self, particles, altLoc='A', model=1, **kwargs):
        """Create an empty scene from particles.
        The Scene object is a wraper of a pandas DataFrame with extra information"""
        pandas.DataFrame.__init__(self, particles)
        # Add metadata dictionary
        self.__dict__['_meta'] = {}

        if 'x' in self.columns:
            if 'y' not in self.columns or 'z' not in self.columns:
                raise TypeError('E')
        elif len(self.columns) == 3:
            pandas.DataFrame.__init__(self, particles, columns=['x', 'y', 'z'])
        else:
            raise TypeError("Incorrect particle format")

        if 'chainID' not in self.columns:
            self['chainID'] = ['A'] * len(self)
        if 'resSeq' not in self.columns:
            self['resSeq'] = [1] * len(self)
        if 'iCode' not in self.columns:
            self['iCode'] = [''] * len(self)
        if 'altLoc' not in self.columns:
            self['altLoc'] = [''] * len(self)
        if 'model' not in self.columns:
            self['model'] = [1] * len(self)
        if 'name' not in self.columns:
            self['name'] = [f'P{i:03}' for i in range(len(self))]
        if 'element' not in self.columns:
            self['element'] = ['C'] * len(self)
        if 'occupancy' not in self.columns:
            self['occupancy'] = [1.0] * len(self)
        if 'tempFactor' not in self.columns:
            self['tempFactor'] = [1.0] * len(self)

        # Map chain index to index
        if 'chain_index' not in self.columns:
            chain_map = {b: a for a, b in enumerate(self['chainID'].unique())}
            self['chain_index'] = self['chainID'].replace(chain_map)

        # Map residue to index
        if 'res_index' not in self.columns:
            resmap = []
            for c, chain in self.groupby('chain_index'):
                residues = (chain['resSeq'].astype(str) + chain['iCode'].astype(str))
                unique_residues = residues.unique()
                dict(zip(unique_residues, range(len(unique_residues))))
                resmap += [residues.replace(dict(zip(unique_residues, range(len(unique_residues)))))]
            self['res_index'] = pandas.concat(resmap)

        # Add metadata
        for attr, value in kwargs.items():
            self._meta[attr] = value

    def select(self, **kwargs):
        index = self.index
        sel = pandas.Series([True] * len(index), index=index)
        for key in kwargs:
            print(key)
            if key == 'altLoc':
                sel &= (self['altLoc'].isin(['', '.'] + kwargs['altLoc']))
            elif key == 'model':
                sel &= (self['model'].isin(kwargs['model']))
            else:
                sel &= (self[key].isin(kwargs[key]))

        # Assert there are not repeated atoms
        index = self[sel][['chain_index', 'res_index', 'name']]
        if len(index.duplicated()) == 0:
            print("Duplicated atoms found")
            print(index[index.duplicated()])
            self._meta['duplicated'] = True

        return Scene(self[sel], **self._meta)

    def split_models(self):
        # TODO: Implement splitting based on model and altLoc.
        # altLoc can be present in multiple regions (1zir)
        pass

    #        for m in self['model'].unique():
    #            for a in sel:
    #                pass

    @classmethod
    def from_pdb(cls, file, **kwargs):
        def pdb_line(line):
            l = dict(recname=str(line[0:6]).strip(),
                     serial=int(line[6:11]),
                     name=str(line[12:16]).strip(),
                     altLoc=str(line[16:17]).strip(),
                     resName=str(line[17:20]).strip(),
                     chainID=str(line[21:22]).strip(),
                     resSeq=int(line[22:26]),
                     iCode=str(line[26:27]).strip(),
                     x=float(line[30:38]),
                     y=float(line[38:46]),
                     z=float(line[46:54]),
                     occupancy=line[54:60].strip(),
                     tempFactor=line[60:66].strip(),
                     element=str(line[76:78]).strip(),
                     charge=str(line[78:80]).strip())
            if l['occupancy'] == '':
                l['occupancy'] = 1.0
            else:
                l['occupancy'] = float(l['occupancy'])
            if l['tempFactor'] == '':
                l['tempFactor'] = 1.0
            else:
                l['tempFactor'] = float(l['tempFactor'])
            if l['charge'] == '':
                l['charge'] = 0.0
            else:
                l['charge'] = float(l['charge'])
            return l

        with open(file, 'r') as pdb:
            lines = []
            mod_lines = []
            model_numbers = []
            model_number = 1
            for i, line in enumerate(pdb):
                if len(line) > 6:
                    header = line[:6]
                    if header == 'ATOM  ' or header == 'HETATM':
                        lines += [pdb_line(line)]
                        model_numbers += [model_number]
                    elif header == "MODRES":
                        m = dict(recname=str(line[0:6]).strip(),
                                 idCode=str(line[7:11]).strip(),
                                 resName=str(line[12:15]).strip(),
                                 chainID=str(line[16:17]).strip(),
                                 resSeq=int(line[18:22]),
                                 iCode=str(line[22:23]).strip(),
                                 stdRes=str(line[24:27]).strip(),
                                 comment=str(line[29:70]).strip())
                        mod_lines += [m]
                    elif header == "MODEL ":
                        model_number = int(line[10:14])
        pdb_atoms = pandas.DataFrame(lines)
        pdb_atoms = pdb_atoms[['recname', 'serial', 'name', 'altLoc',
                               'resName', 'chainID', 'resSeq', 'iCode',
                               'x', 'y', 'z', 'occupancy', 'tempFactor',
                               'element', 'charge']]
        pdb_atoms['model'] = model_numbers

        if len(mod_lines) > 0:
            kwargs.update(dict(modified_residues=pandas.DataFrame(mod_lines)))

        return cls(pdb_atoms, **kwargs)

    @classmethod
    def from_cif(cls, file, **kwargs):
        _cif_pdb_rename = {'id': 'serial',
                           'label_atom_id': 'name',
                           'label_alt_id': 'altLoc',
                           'label_comp_id': 'resName',
                           'label_asym_id': 'chainID',
                           'label_seq_id': 'resSeq',
                           'pdbx_PDB_ins_code': 'iCode',
                           'Cartn_x': 'x',
                           'Cartn_y': 'y',
                           'Cartn_z': 'z',
                           'occupancy': 'occupancy',
                           'B_iso_or_equiv': 'tempFactor',
                           'type_symbol': 'element',
                           'pdbx_formal_charge': 'charge',
                           'pdbx_PDB_model_num': 'model'}

        data = []
        with open(file) as f:
            reader = utils.PdbxReader(f)
            reader.read(data)
        block = data[0]
        atom_data = block.getObj('atom_site')
        cif_atoms = pandas.DataFrame([atom_data.getFullRow(i) for i in range(atom_data.getRowCount())],
                                     columns=atom_data.getAttributeList(),
                                     index=range(atom_data.getRowCount()))
        # Rename columns to pdb convention
        cif_atoms = cif_atoms.rename(_cif_pdb_rename, axis=1)
        for col in cif_atoms.columns:
            try:
                cif_atoms[col] = cif_atoms[col].astype(float)
                if ((cif_atoms[col].astype(int) - cif_atoms[col]) ** 2).sum() == 0:
                    cif_atoms[col] = cif_atoms[col].astype(int)
                continue
            except ValueError:
                pass
        return cls(cif_atoms, **kwargs)

    @classmethod
    def from_gro(cls, gro, **kwargs):
        """Not implemented"""
        return cls()

    @classmethod
    def from_fixPDB(cls, filename=None, pdbfile=None, pdbxfile=None, url=None, pdbid=None,
                    **kwargs):
        """Uses the pdbfixer library to fix a pdb file, replacing non standard residues, removing
        hetero-atoms and adding missing hydrogens. The input is a pdb file location,
        the output is a fixer object, which is a pdb in the openawsem format."""
        import pdbfixer
        fixer = pdbfixer.PDBFixer(filename=filename, pdbfile=pdbfile, pdbxfile=pdbxfile, url=url, pdbid=pdbid)
        fixer.findMissingResidues()
        chains = list(fixer.topology.chains())
        keys = fixer.missingResidues.keys()
        for key in list(keys):
            chain_tmp = chains[key[0]]
            if key[1] == 0 or key[1] == len(list(chain_tmp.residues())):
                del fixer.missingResidues[key]

        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.removeHeterogens(keepWater=False)
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()  # Warning: importing 'simtk.openmm' is deprecated.  Import 'openmm' instead.
        fixer.addMissingHydrogens(7.0)

        pdb = fixer
        """ Parses a pdb in the openmm format and outputs a table that contains all the information
        on a pdb file """
        cols = ['recname', 'serial', 'name', 'altLoc',
                'resName', 'chainID', 'resSeq', 'iCode',
                'x', 'y', 'z', 'occupancy', 'tempFactor',
                'element', 'charge']
        data = []

        for atom, pos in zip(pdb.topology.atoms(), pdb.positions):
            residue = atom.residue
            chain = residue.chain
            pos = pos.value_in_unit(pdbfixer.pdbfixer.unit.angstrom)
            data += [dict(zip(cols, ['ATOM', int(atom.id), atom.name, '',
                                     residue.name, chain.id, int(residue.id), '',
                                     pos[0], pos[1], pos[2], 0, 0,
                                     atom.element.symbol, '']))]
        atom_list = pandas.DataFrame(data)
        atom_list = atom_list[cols]
        atom_list.index = atom_list['serial']
        return cls(atom_list, **kwargs)

    @classmethod
    def concatenate(cls, scene_list):
        #Set chain names
        chainID = []
        name_generator = utils.chain_name_generator()
        for scene in scene_list:
            if 'chainID' not in scene:
                chainID += [next(name_generator)]*len(scene)
            else:
                chains = list(scene['chainID'].unique())
                chains.sort()
                chain_replace = {chain: next(name_generator) for chain in chains}
                chainID += list(scene['chainID'].replace(chain_replace))
        name_generator.close()
        model = pandas.concat(scene_list)
        model['chainID'] = chainID
        model.index = range(len(model))
        return cls(model)

    # Writing
    def write_pdb(self, file_name=None, verbose=False):

        # TODO Add connectivity output
        # Fill empty columns
        if verbose:
            print(f"Writing pdb file ({len(self)} atoms): {file_name}")

        pdb_table = self.copy()
        pdb_table['serial'] = np.arange(1, len(self) + 1) if 'serial' not in pdb_table else pdb_table['serial']
        pdb_table['name'] = 'A' if 'name' not in pdb_table else pdb_table['name']
        pdb_table['altLoc'] = '' if 'altLoc' not in pdb_table else pdb_table['altLoc']
        pdb_table['resName'] = 'R' if 'resName' not in pdb_table else pdb_table['resName']
        pdb_table['chainID'] = 'C' if 'chainID' not in pdb_table else pdb_table['chainID']
        pdb_table['resSeq'] = 1 if 'resSeq' not in pdb_table else pdb_table['resSeq']
        pdb_table['iCode'] = '' if 'iCode' not in pdb_table else pdb_table['iCode']
        assert 'x' in pdb_table.columns, 'Coordinate x not in particle definition'
        assert 'y' in pdb_table.columns, 'Coordinate x not in particle definition'
        assert 'z' in pdb_table.columns, 'Coordinate x not in particle definition'
        pdb_table['occupancy'] = 0 if 'occupancy' not in pdb_table else pdb_table['occupancy']
        pdb_table['tempFactor'] = 0 if 'tempFactor' not in pdb_table else pdb_table['tempFactor']
        pdb_table['element'] = '' if 'element' not in pdb_table else pdb_table['element']
        pdb_table['charge'] = 0 if 'charge' not in pdb_table else pdb_table['charge']

        # Override chain names if molecule is present
        if 'molecule' in pdb_table:
            import string
            cc = utils.chain_name_generator(format='pdb')
            molecules = self.atoms['molecule'].unique()
            cc_d = dict(zip(molecules, cc))
            # cc_d = dict(zip(range(1, len(cc) + 1), cc))
            pdb_table['chainID'] = self.atoms['molecule'].replace(cc_d)

        # Write pdb file
        lines = ''
        for i, atom in pdb_table.iterrows():
            line = f'ATOM  {i:>5} {atom["name"]:^4} {atom["resName"]:<3} {atom["chainID"]}{atom["resSeq"]:>4}' + \
                   '    ' + \
                   f'{atom.x:>8.3f}{atom.y:>8.3f}{atom.z:>8.3f}' + ' ' * 22 + f'{atom.element:2}' + ' ' * 2
            assert len(line) == 80, f'An item in the atom table is longer than expected\n{line}'
            lines += line + '\n'

        if file_name is None:
            return io.StringIO(lines)
        else:
            with open(file_name, 'w+') as out:
                out.write(lines)

    def write_cif(self, file_name=None, verbose=False):
        """Write a PDBx/mmCIF file.

        Parameters
        ----------
        topology : Topology
            The Topology defining the molecular system being written
        file : file=stdout
            A file to write the file to
        entry : str=None
            The entry ID to assign to the CIF file
        keepIds : bool=False
            If True, keep the residue and chain IDs specified in the Topology
            rather than generating new ones.  Warning: It is up to the caller to
            make sure these are valid IDs that satisfy the requirements of the
            PDBx/mmCIF format.  Otherwise, the output file will be invalid.
        """
        """Write out a model to a PDBx/mmCIF file.

        Parameters
        ----------
        topology : Topology
            The Topology defining the model to write
        positions : list
            The list of atomic positions to write
        file : file=stdout
            A file to write the model to
        modelIndex : int=1
            The model number of this frame
        keepIds : bool=False
            If True, keep the residue and chain IDs specified in the Topology
            rather than generating new ones.  Warning: It is up to the caller to
            make sure these are valid IDs that satisfy the requirements of the
            PDBx/mmCIF format.  Otherwise, the output file will be invalid.
        """
        # TODO Add connectivity output
        if verbose:
            print(f"Writing cif file ({len(self)} atoms): {file_name}")

        # Fill empty columns
        pdbx_table = self.copy()
        pdbx_table['serial'] = np.arange(1, len(self) + 1) if 'serial' not in pdbx_table else pdbx_table['serial']
        pdbx_table['name'] = 'A' if 'name' not in pdbx_table else pdbx_table['name'].str.strip().replace('', '.')
        pdbx_table['altLoc'] = '?' if 'altLoc' not in pdbx_table else pdbx_table['altLoc'].str.strip().replace('', '.')
        pdbx_table['resName'] = 'R' if 'resName' not in pdbx_table else pdbx_table['resName'].str.strip().replace('',
                                                                                                                  '.')
        pdbx_table['chainID'] = 'C' if 'chainID' not in pdbx_table else pdbx_table['chainID'].str.strip().replace('',
                                                                                                                  '.')
        pdbx_table['resSeq'] = 1 if 'resSeq' not in pdbx_table else pdbx_table['resSeq']
        pdbx_table['resIC'] = 1 if 'resIC' not in pdbx_table else pdbx_table['resIC']
        pdbx_table['iCode'] = '' if 'iCode' not in pdbx_table else pdbx_table['iCode'].str.strip().replace('', '.')
        assert 'x' in pdbx_table.columns, 'Coordinate x not in particle definition'
        assert 'y' in pdbx_table.columns, 'Coordinate x not in particle definition'
        assert 'z' in pdbx_table.columns, 'Coordinate x not in particle definition'
        pdbx_table['occupancy'] = 0 if 'occupancy' not in pdbx_table else pdbx_table['occupancy']
        pdbx_table['tempFactor'] = 0 if 'tempFactor' not in pdbx_table else pdbx_table['tempFactor']
        pdbx_table['element'] = 'C' if 'element' not in pdbx_table else pdbx_table['element'].str.strip().replace('',
                                                                                                                  '.')
        pdbx_table['charge'] = 0 if 'charge' not in pdbx_table else pdbx_table['charge']
        pdbx_table['model'] = 0 if 'model' not in pdbx_table else pdbx_table['model']

        lines = ""
        lines += "data_pdbx\n"
        lines += "#\n"
        lines += "loop_\n"
        lines += "_atom_site.group_PDB\n"
        lines += "_atom_site.id\n"
        lines += "_atom_site.label_atom_id\n"
        lines += "_atom_site.label_comp_id\n"
        lines += "_atom_site.label_asym_id\n"
        lines += "_atom_site.label_seq_id\n"
        lines += "_atom_site.label_alt_id\n"
        lines += "_atom_site.auth_atom_id\n"
        lines += "_atom_site.auth_comp_id\n"
        lines += "_atom_site.auth_asym_id\n"
        lines += "_atom_site.auth_seq_id\n"
        lines += "_atom_site.pdbx_PDB_ins_code\n"
        lines += "_atom_site.Cartn_x\n"
        lines += "_atom_site.Cartn_y\n"
        lines += "_atom_site.Cartn_z\n"
        lines += "_atom_site.occupancy\n"
        lines += "_atom_site.B_iso_or_equiv\n"
        lines += "_atom_site.type_symbol\n"
        lines += "_atom_site.pdbx_formal_chrge\n"
        lines += "_atom_site.pdbx_PDB_model_num\n"

        pdbx_table['line'] = 'ATOM'
        for col in ['serial',
                    'name', 'resName', 'chainID', 'resSeq', 'iCode',
                    'name', 'resName', 'chainID', 'resSeq','iCode',
                    'x', 'y', 'z',
                    'occupancy', 'tempFactor',
                    'element', 'charge', 'model']:
            pdbx_table['line'] += " "
            pdbx_table['line'] += pdbx_table[col].astype(str)
        pdbx_table['line'] += '\n'
        lines += ''.join(pdbx_table['line'])
        lines += '#\n'

        if file_name is None:
            return io.StringIO(lines)
        else:
            with open(file_name, 'w+') as out:
                out.write(lines)

    def write_gro(self, file_name, box_size=None, verbose=False):
        if verbose:
            print(f"Writing pdb file ({len(self)} atoms): {file_name}")

        gro_line = "%5d%-5s%5s%5d%8s%8s%8s%8s%8s%8s\n"
        pdb_atoms = self.copy()
        pdb_atoms['resName'] = pdb_atoms[
            'resName']  # self.atoms['molecule_name'].replace({'actin':'ACT','camkii':'CAM'})
        # pdb_atoms['name']       = self.atoms['type'].replace({1:'Aa',2:'Ab',3:'Ca',4:'Cb',5:'Da',6:'Db'})
        pdb_atoms['serial'] = np.arange(1, len(self) + 1)
        pdb_atoms['chainID'] = pdb_atoms['molecule']
        if box_size is None:
            box_size = pdb_atoms[['x', 'y', 'z']].max().max()

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

    # get methods
    def get_coordinates(self):
        return self[['x', 'y', 'z']]

    def get_sequence(self):
        pass

    def set_coordinates(self, coordinates):
        self[['x', 'y', 'z']] = coordinates

    def copy(self, deep=True):
        return Scene(super().copy(deep), **self._meta)

    def correct_modified_aminoacids(self):
        out = self.copy()
        if 'modified_residues' in self._meta:
            for i, row in out.modified_residues.iterrows():
                sel = ((out['resName'] == row['resName']) &
                       (out['chainID'] == row['chainID']) &
                       (out['resSeq'] == row['resSeq']))
                out.loc[sel, 'resName'] = row['stdRes']
        return out

    def rotate(self, rotation_matrix):
        return self.dot(rotation_matrix)

    def translate(self, other):
        new = self.copy()
        new.at[:, ['x', 'y', 'z']] = self.get_coordinates() + other
        return new

    def dot(self, other):
        new = self.copy()
        new.at[:, ['x', 'y', 'z']] = self.get_coordinates().dot(other)
        return new

    # Container operations
    # No container operations, a subset may require some coordinates and not the chain index
    # def __getitem__(self, key):
    # try:
    #    return Scene(super().__getitem__(key))
    # except TypeError:
    #    return super().__getitem__(key)
    # Built ins
    def __repr__(self):
        return f'<Scene ({len(self)} particles)>\n{super().__repr__()}'

    def __add__(self, other):
        if isinstance(other, self.__class__):
            concatenated = pandas.concat([self, other], axis=0)
            return Scene(concatenated)
        else:
            coord = self.get_coordinates()
            temp = self.copy()
            temp.set_coordinates(coord + other)
            return temp

    def __radd__(self, other):
        if isinstance(other, self.__class__):
            return other + s
        else:
            return Scene(super().__add__(other))

    def __mul__(self, other):
        coord = self.get_coordinates()
        temp = self.copy()
        temp.set_coordinates(coord * other)
        return temp

    def __rmul__(self, other):
        coord = self.get_coordinates()
        temp = self.copy()
        temp.set_coordinates(other * coord)
        return temp

    def __getattr__(self, attr):
        if '_meta' in self.__dict__ and attr in self._meta:
            return self._meta[attr]
        else:
            raise AttributeError(f"type object {str(self.__class__)} has no attribute {str(attr)}")

    def __setattr__(self, attr, value):
        self._meta[attr] = value

    """
    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)

        elif 'particles' in self.__dict__:
            return getattr(self.particles, attr)
        else:
            if '__repr__' in self.__dict__:
                raise AttributeError(f"type object {str(self)} has no attribute {str(attr)}")
            else:
                raise AttributeError()

    def __setattr__(self, attr, value):
        if 'particles' in self.__dict__ and attr in self.particles.__dict__:
            return setattr(self.particles, attr, value)
        else:
            self.__dict__[attr] = value
    """



if __name__ == '__main__':
    particles = pandas.DataFrame([[0, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, 1]],
                                 columns=['x', 'y', 'z'])
    s = Scene(particles)
    s.write_pdb('test.pdb')

    s = Scene.from_pdb('test.pdb')

    s.write_cif('test.cif')

    s = Scene.from_cif('test.cif')

    s = Scene.from_fixPDB(pdbid='1JGE')

    s1 = Scene(particles)
    s1.write_pdb('test.pdb')
    s2 = Scene.from_pdb('test.pdb')
    s2.write_cif('test.cif')
    s3 = Scene.from_cif('test.cif')
    s3.write_pdb('test2.pdb')
    s4 = Scene.from_pdb('test2.pdb')

    s1.to_csv('particles_1.csv')
    s2.to_csv('particles_2.csv')
    s3.to_csv('particles_3.csv')
    s4.to_csv('particles_4.csv')

"""
import numpy as np
import pandas as pd

def h5store(filename, df, **kwargs):
    store = pd.HDFStore(filename)
    store.put('mydata', df)
    store.get_storer('mydata').attrs.metadata = kwargs
    store.close()

def h5load(store):
    data = store['mydata']
    metadata = store.get_storer('mydata').attrs.metadata
    return data, metadata

a = pd.DataFrame(
    data=pd.np.random.randint(0, 100, (10, 5)), columns=list('ABCED'))

filename = '/tmp/data.h5'
metadata = dict(local_tz='US/Eastern')
h5store(filename, a, **metadata)
with pd.HDFStore(filename) as store:
    data, metadata = h5load(store)

print(data)
#     A   B   C   E   D
# 0   9  20  92  43  25
# 1   2  64  54   0  63
# 2  22  42   3  83  81
# 3   3  71  17  64  53
# 4  52  10  41  22  43
# 5  48  85  96  72  88
# 6  10  47   2  10  78
# 7  30  80   3  59  16
# 8  13  52  98  79  65
# 9   6  93  55  40   3

$DATE$ $TIME$
"""
