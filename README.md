# parallel mordred generator

## Requirement
```
conda install mpi4py
conda install -c rdkit -c mordred-descriptor mordred
conda install rdkit
conda install pytables
```

# Run
```
# 1 master, 4 workers
mpiexec -n 5 python run.py
```

## Parameters
```
usage: run.py [-h] [--batch_size BATCH_SIZE] [--smiles SMILES]
              [--format {csv,tsv,hdf5}]

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        Batch size
  --smiles SMILES       Input Smile path
  --format {csv,tsv,hdf5}
                        Dataframe file format. Default hdf5
```

# Running on Theta
## Create a conda env
```
module load miniconda-3.6/conda-4.5.12

conda create python=3.5 -p /projects/CSC249ADOA01/candle_aesp/conda
conda activate /projects/CSC249ADOA01/candle_aesp/conda

conda install -c conda-forge rdkit
conda install -c rdkit -c mordred-descriptor mordred
conda install pytables
```

## Runtime setting
```
unset PYTHONPATH
unset LD_LIBRARY_PATH

module load datascience/mpi4py

export PYTHONPATH=$PYTHONPATH:/projects/CSC249ADOA01/candle_aesp/conda/lib/python3.5/site-packages/
export LD_LIBRARY_PATH=/projects/CSC249ADOA01/candle_aesp/conda/lib/:$LD_LIBRARY_PATH
```

## Run
```
aprun -n 10 -N 1 -cc none python run.py
```