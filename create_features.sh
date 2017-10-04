#!/usr/bin/env bash
qsub -cwd -l mem=32G,time=1:30: -o /ifs/scratch/c2b2/ip_lab/ab4377/test_output.txt -e /ifs/scratch/c2b2/ip_lab/ab4377/error.txt -j y create_feature_job.sh "additional" "outbound" 0 6500 7500
qsub -cwd -l mem=32G,time=1:30: -o /ifs/scratch/c2b2/ip_lab/ab4377/test_output.txt -e /ifs/scratch/c2b2/ip_lab/ab4377/error.txt -j y create_feature_job.sh "additional" "outbound" 1 6500 7500
qsub -cwd -l mem=32G,time=1:30: -o /ifs/scratch/c2b2/ip_lab/ab4377/test_output.txt -e /ifs/scratch/c2b2/ip_lab/ab4377/error.txt -j y create_feature_job.sh "additional" "outbound" 2 6500 7500
qsub -cwd -l mem=32G,time=1:30: -o /ifs/scratch/c2b2/ip_lab/ab4377/test_output.txt -e /ifs/scratch/c2b2/ip_lab/ab4377/error.txt -j y create_feature_job.sh "additional" "outbound" 3 6500 7500