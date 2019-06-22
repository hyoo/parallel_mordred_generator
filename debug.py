from mpi4py import MPI
from rdkit import Chem
from mordred import Calculator, descriptors
from enum import IntEnum
import math
import argparse
import socket
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
    def __init__(self, rank=None, args=None):
        self.calc = Calculator(descriptors, ignore_3D=True)
        if rank is not None:
            self.rank = rank
        else:
            raise ValueError('rank is not set properly')
        if args is not None:
            self.args = args
        else:
            raise ValueError('args is not set properly')

    def do(self, data):
        if self.args.verbose:
            print('rank {} received data: {} rows'.format(self.rank, len(data)))

        if self.args.echo:
            df = pd.DataFrame(data, columns=['SMILE'])
        else:
            mols = [Chem.MolFromSmiles(smi) for smi in data]
            df = self.calc.pandas(mols, quiet=True)
            df.fill_missing(inplace=True)
            df.insert(0, 'SMILE', data)

        if self.args.verbose:
            print('rank {} generated data: {}'.format(self.rank, len(df)))
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
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='verbose')
    parser.add_argument('--echo', action='store_true',
                        help='does not generate. just echo input data')

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
    if args.verbose:
        print('master node initialized at {}'.format(socket.gethostname()))

    for i in range(1, size):
        anext = smiles.get_next()
        if not anext:
            break
        comm.send(obj=anext, dest=i, tag=Tags.CONTINUE)

    df = None
    while 1:
        anext = smiles.get_next()
        if not anext:
            break
        data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        df = data if df is None else df.append(data, ignore_index=True, sort=False)
        comm.send(obj=anext, dest=status.Get_source(), tag=Tags.CONTINUE)

    for i in range(1, size):
        data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
        df = df.append(data, ignore_index=True, sort=False)

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
    rank = comm.Get_rank()
    worker = Worker(rank, args)

    if args.verbose:
        print('slave rank: {} initialized at {}'.format(rank, socket.gethostname()))

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
