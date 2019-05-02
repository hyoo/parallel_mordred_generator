from mpi4py import MPI
from rdkit import Chem
from mordred import Calculator, descriptors
from enum import IntEnum
import math


class WorkOrders():
    def __init__(self, orders):
        self.orders = orders[:]
        self.batch_size = 10
        self.idx = -1
        self.total = len(orders)
        self.steps = math.ceil(self.total / self.batch_size)

    def get_next(self):
        if self.idx > self.steps:
            return None
        self.idx += 1
        return self.orders[self.idx * self.batch_size:(self.idx + 1) * self.batch_size] if self.idx < self.steps else self.orders[self.idx * self.batch_size: self.total]


class Worker():
    def __init__(self):
        self.calc = Calculator(descriptors, ignore_3D=True)

    def do(self, data):
        mols = [Chem.MolFromSmiles(smi) for smi in data]
        df = self.calc.pandas(mols)
        df.fill_missing(inplace=True)
        df['SMILE'] = data
        return df


class WordCounter():
    def do(self, data):
        return "{},{}".format(data, len(data))


Tags = IntEnum('Tags', 'CONTINUE EXIT')


def master():
    smiles = []
    with open('smiles.txt', 'r') as input_lines:
        for line in input_lines:
            smiles.append(line.strip())

    smiles = WorkOrders(smiles)

    # MPI
    size = MPI.COMM_WORLD.Get_size()
    comm = MPI.COMM_WORLD
    status = MPI.Status()

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
        df = data if df is None else df.append(data, ignore_index=True)
        comm.send(obj=anext, dest=status.Get_source(), tag=Tags.CONTINUE)

    for i in range(1, size):
        data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
        df = df.append(data, ignore_index=True)

    # terminate slaves
    for i in range(1, size):
        comm.send(obj=None, dest=i, tag=Tags.EXIT)

    print("Generated {} descriptors".format(len(df)))
    df.to_feather('mordred.feather')


def slave():
    comm = MPI.COMM_WORLD
    status = MPI.Status()
    # worker = WordCounter()
    worker = Worker()

    while 1:
        data = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        if status.Get_tag() == Tags.EXIT:
            break
        comm.send(obj=worker.do(data), dest=0)


if __name__ == '__main__':
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()

    if rank == 0:
        master()
    else:
        slave()
