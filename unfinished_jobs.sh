#!/usr/bin/env bash
dataset_type=$1
number_of_records=$2
category_index=$3
batch_size=500
number_of_batches=$((number_of_records/batch_size))

i=0
while [[ $i -lt $number_of_batches ]] ; do
    start_index=$((i*batch_size))
    end_index=$(( (i+1)*batch_size))
    qsub -cwd -l mem=32G,time=1:30: -o /ifs/scratch/c2b2/ip_lab/ab4377/test_output.txt -e /ifs/scratch/c2b2/ip_lab/ab4377/error.txt -j y create_feature_job.sh $dataset_type "outbound" $category_index $start_index $end_index
    #echo $i
    i=$((i+1))
    global_count=$((global_count+1))
done
start_index=$((i*batch_size))
end_index=$number_of_records
if [[ $start_index -lt $end_index ]]
then
    qsub -cwd -l mem=32G,time=1:30: -o /ifs/scratch/c2b2/ip_lab/ab4377/test_output.txt -e /ifs/scratch/c2b2/ip_lab/ab4377/error.txt -j y create_feature_job.sh $dataset_type "outbound" $category_index $start_index $end_index
fi

echo $global_count