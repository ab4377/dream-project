#!/usr/bin/env bash
cd /ifs/scratch/c2b2/ip_lab/ab4377/dream/
module load python/2.7.11
export PYTHONPATH=${PYTHONPATH}:/ifs/scratch/c2b2/ip_lab/ab4377/dream/pyhsmm/
source bin/activate
python RunOnData.py "$1" "$2" "$3" "$4" "$5"