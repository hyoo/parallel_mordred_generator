from rdkit import Chem
from mordred import Calculator, descriptors
from enum import IntEnum
import math
import argparse


class WorkOrders():
    def __init__(self, orders, args):
        self.orders = orders[:]
        self.batch_size = args.batch_size
        self.idx = -1
        self.total = len(orders)
        self.steps = math.ceil(self.total / self.batch_size)

    def get_next(self):
        if self.idx > self.steps:
            return None
        self.idx += 1
        if self.idx < self.steps:
            return self.orders[self.idx * self.batch_size: (self.idx + 1) * self.batch_size]
        else:
            return self.orders[self.idx * self.batch_size: self.total]


class Worker():
    def __init__(self, args):
        self.calc = Calculator(descriptors, ignore_3D=True)
        self.args = args

    def gen(self, smile):
        mol = Chem.MolFromSmiles(smile)
        desc = self.calc(mol).fill_missing().asdict()
        desc['SMILE'] = smile
        return desc

    def do(self, data):
        return [self.gen(smile) for smile in data]


class ParallelWorker(Worker):
    def gen(self, rs, smile):
        desc = rs.fill_missing().asdict()
        desc['SMILE'] = smile
        return desc

    def do(self, data):
        mols = [Chem.MolFromSmiles(smile) for smile in data]
        iterResults = self.calc.map(mols, nproc=self.args.nproc, quiet=True)
        return [self.gen(rs, data[i]) for i, rs in enumerate(iterResults)]


class PandasWorker(Worker):
    def do(self, data):
        mols = [Chem.MolFromSmiles(smi) for smi in data]
        df = self.calc.pandas(mols, nproc=self.args.nproc, quiet=True)
        df.fill_missing(inplace=True)
        df.insert(0, 'SMILE', data)
        return df


Tags = IntEnum('Tags', 'CONTINUE EXIT')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=10,
                        help='Batch size')
    parser.add_argument('--smiles', type=str, default='smiles.txt',
                        help='Input Smile path')
    parser.add_argument('--format', default='hdf5',
                        choices=['csv', 'tsv', 'hdf5'],
                        help='Dataframe file format. Default hdf5')
    parser.add_argument('--nproc', type=int, default=None,
                        help='number of concurrent generator processes')

    args, unparsed = parser.parse_known_args()
    return args, unparsed
