#!/bin/bash 
#SBATCH --nodes=2 --time=0:10:00  
#SBATCH -C gpu 
#SBATCH --account nvendor_g
#SBATCH --tasks-per-node 4
#SBATCH --cpus-per-task 32
#SBATCH --gpus-per-task 1
#SBATCH -J multinode
#SBATCH -o %x-%j.out

DATADIR=/pscratch/sd/j/joshr/nbody2hydro/datacopies
LOGDIR=/pscratch/sd/j/joshr/ampUNet/logs

hostname

#~/dummy

srun -u shifter --image=romerojosh/containers:sc21_tutorial --module=gpu \
    -V ${DATADIR}:/data -V ${LOGDIR}:/logs \
    bash -c '
    source export_DDP_vars.sh
    python train.py --config=A100_crop64_8GPU --data_loader_config dali-lowmem
    '

