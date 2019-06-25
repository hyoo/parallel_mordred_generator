from mpi4py import MPI
from rdkit import Chem
from mordred import Calculator, descriptors
from enum import IntEnum
import math
import argparse
import pandas as pd


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
        iterResults = self.calc.map(mols, nproc=args.nproc, quiet=True)
        return [self.gen(rs, data[i]) for i, rs in enumerate(iterResults)]


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


def master(args):
    smiles = []
    with open(args.smiles, 'r') as input_lines:
        for line in input_lines:
            smiles.append(line.strip())

    smiles = WorkOrders(smiles, args)

    # MPI
    size = MPI.COMM_WORLD.Get_size()
    comm = MPI.COMM_WORLD
    status = MPI.Status()

    for i in range(1, size):
        anext = smiles.get_next()
        if not anext:
            break
        comm.send(obj=anext, dest=i, tag=Tags.CONTINUE)

    df = pd.DataFrame()
    while 1:
        anext = smiles.get_next()
        if not anext:
            break
        data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        df = df.append(pd.DataFrame(data), ignore_index=True)
        comm.send(obj=anext, dest=status.Get_source(), tag=Tags.CONTINUE)

    for i in range(1, size):
        data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
        df = df.append(pd.DataFrame(data), ignore_index=True)

    # terminate slaves
    for i in range(1, size):
        comm.send(obj=None, dest=i, tag=Tags.EXIT)

    print("Generated {} descriptors".format(len(df)))

    # save
    if args.format == 'csv':
        df.to_csv('mordred.csv', float_format='%g', index=False)
    elif args.format == 'tsv':
        df.to_csv('mordred.tsv', sep='\t', float_format='%g', index=False)
    else:
        df.to_hdf('mordred.h5', key='df', mode='w', complib='blosc:snappy', complevel=9)


def slave(args):
    comm = MPI.COMM_WORLD
    status = MPI.Status()
    # worker = Worker(args)
    worker = ParallelWorker(args)

    while 1:
        data = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        if status.Get_tag() == Tags.EXIT:
            break
        comm.send(obj=worker.do(data), dest=0)


if __name__ == '__main__':
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    args, unparsed = parse_arguments()

    if rank == 0:
        master(args)
    else:
        slave(args)
