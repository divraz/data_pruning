#!/bin/bash
 
#SBATCH -n 4                       # number of cores
#SBATCH -N 1
#SBATCH -t 0-24:40                  # wall time (D-HH:MM)
#SBATCH -p cidsegpu1
#SBATCH -q cidsegpu1_contrib_res 
#SBATCH --gres=gpu:1
##SBATCH -A draj5             # Account hours will be pulled from (commented out with double # in front)
#SBATCH -o logs/%j.out             # STDOUT (%j = JobId)
#SBATCH -e logs/%j.err             # STDERR (%j = JobId)
#SBATCH --mail-type=ALL             # Send a notification when the job starts, stops, or fails
#SBATCH --mail-user=draj5@asu.edu # send-to address

#module purge    # Always purge modules to ensure a consistent environment


#module load anaconda3/5.3.0
#module load cudnn/7.0
#source activate temp1

cd /home/draj5/projects/data_pruning/experiments/sst2/
python $1 $2
