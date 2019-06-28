from mpi4py import MPI
import pandas as pd
from generator import WorkOrders, ParallelWorker
from generator import parse_arguments, Tags


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
