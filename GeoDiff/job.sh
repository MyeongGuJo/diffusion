#!/bin/bash

#SBATCH -J geodiff  # Job name
#SBATCH -o out/geodiff_%j.out       # Name of stdout output file (%j expands to jobId)

srun python test.py