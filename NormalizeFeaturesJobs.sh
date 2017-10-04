#!/usr/bin/env bash
qsub -cwd -l mem=32G,time=1:30: -o /ifs/scratch/c2b2/ip_lab/ab4377/test_output.txt -e /ifs/scratch/c2b2/ip_lab/ab4377/error.txt -j y NormalizeFeatures.sh "outbound" 0
qsub -cwd -l mem=32G,time=1:30: -o /ifs/scratch/c2b2/ip_lab/ab4377/test_output.txt -e /ifs/scratch/c2b2/ip_lab/ab4377/error.txt -j y NormalizeFeatures.sh "outbound" 1
qsub -cwd -l mem=32G,time=1:30: -o /ifs/scratch/c2b2/ip_lab/ab4377/test_output.txt -e /ifs/scratch/c2b2/ip_lab/ab4377/error.txt -j y NormalizeFeatures.sh "outbound" 2
qsub -cwd -l mem=32G,time=1:30: -o /ifs/scratch/c2b2/ip_lab/ab4377/test_output.txt -e /ifs/scratch/c2b2/ip_lab/ab4377/error.txt -j y NormalizeFeatures.sh "outbound" 3
