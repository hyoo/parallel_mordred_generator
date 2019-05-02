from mpi4py import MPI
from rdkit import Chem
from mordred import Calculator, descriptors
from enum import IntEnum


class WorkOrders():
    def __init__(self, orders):
        self.orders = orders[:]

    def get_next(self):
        if len(self.orders) == 0:
            return None
        return self.orders.pop()


class Worker():
    def __init__(self):
        self.calc = Calculator(descriptors, ignore_3D=True)

    def do(self, data):
        return self.calc(Chem.MolFromSmiles(data))


class WordCounter():
    def do(self, data):
        return "{},{}".format(data, len(data))


Tags = IntEnum('Tags', 'CONTINUE EXIT')

def master():
    all_data = []
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

    while 1:
        anext = smiles.get_next()
        if not anext:
            break
        data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        all_data.append(data)
        comm.send(obj=anext, dest=status.Get_source(), tag=Tags.CONTINUE)

    for i in range(1, size):
        data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
        all_data.append(data)

    for i in range(1, size):
        comm.send(obj=None, dest=i, tag=Tags.EXIT)

    print(all_data, len(all_data))


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
