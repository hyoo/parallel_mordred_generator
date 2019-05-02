# parallel mordred generator

## Requirement
```
conda install mpi4py
conda install -c rdkit -c mordred-descriptor mordred
conda install rdkit
```

# Run
```
# 1 master, 4 workers
mpiexec -n 5 python run.py
```
