#!/usr/bin/env bash
dataset_type=$1
number_of_records=$2
batch_size=500
number_of_batches=$((number_of_records/batch_size))
echo $number_of_batches
#outbound jobs
global_count=0
c=0
while [[ $c -lt 4 ]] ; do
    i=0
    while [[ $i -lt $number_of_batches ]] ; do
        start_index=$((i*batch_size))
        end_index=$(( (i+1)*batch_size))
        qsub -cwd -l mem=16G,time=:45: -o /ifs/scratch/c2b2/ip_lab/ab4377/test_output.txt -e /ifs/scratch/c2b2/ip_lab/ab4377/error.txt -j y create_feature_job.sh $dataset_type "outbound" $c $start_index $end_index
        i=$((i+1))
        global_count=$((global_count+1))
    done
    start_index=$((i*batch_size))
    end_index=$number_of_records
    if [[ $start_index -lt $end_index ]]
    then
        qsub -cwd -l mem=16G,time=:45: -o /ifs/scratch/c2b2/ip_lab/ab4377/test_output.txt -e /ifs/scratch/c2b2/ip_lab/ab4377/error.txt -j y create_feature_job.sh $dataset_type "outbound" $c $start_index $end_index
    fi
    c=$((c+1))
done

#return jobs
c=0
while [[ $c -lt 4 ]] ; do
    i=0
    while [[ $i -lt $number_of_batches ]] ; do
        start_index=$((i*batch_size))
        end_index=$(( (i+1)*batch_size))
        qsub -cwd -l mem=16G,time=:45: -o /ifs/scratch/c2b2/ip_lab/ab4377/test_output.txt -e /ifs/scratch/c2b2/ip_lab/ab4377/error.txt -j y create_feature_job.sh $dataset_type "return" $c $start_index $end_index
        i=$((i+1))
        global_count=$((global_count+1))
    done
    start_index=$((i*batch_size))
    end_index=$number_of_records
    if [[ $start_index -lt $end_index ]]
    then
        qsub -cwd -l mem=16G,time=:45: -o /ifs/scratch/c2b2/ip_lab/ab4377/test_output.txt -e /ifs/scratch/c2b2/ip_lab/ab4377/error.txt -j y create_feature_job.sh $dataset_type "return" $c $start_index $end_index
    fi
    c=$((c+1))
done

#outbound+return jobs
c=0
while [[ $c -lt 4 ]] ; do
    i=0
    while [[ $i -lt $number_of_batches ]] ; do
        start_index=$((i*batch_size))
        end_index=$(( (i+1)*batch_size))
        qsub -cwd -l mem=64G,time=5:30: -o /ifs/scratch/c2b2/ip_lab/ab4377/test_output.txt -e /ifs/scratch/c2b2/ip_lab/ab4377/error.txt -j y create_feature_job.sh $dataset_type "both" $c $start_index $end_index
        i=$((i+1))
        global_count=$((global_count+1))
    done
    start_index=$((i*batch_size))
    end_index=$number_of_records
    if [[ $start_index -lt $end_index ]]
    then
        qsub -cwd -l mem=64G,time=5:30: -o /ifs/scratch/c2b2/ip_lab/ab4377/test_output.txt -e /ifs/scratch/c2b2/ip_lab/ab4377/error.txt -j y create_feature_job.sh $dataset_type "both" $c $start_index $end_index
    fi
    c=$((c+1))
done
echo $global_count