base_path="/ifs/home/c2b2/ip_lab/shares/DATA/fwd_bwd_data/converted_fb_accel_data/"
dest_file="/ifs/home/c2b2/ip_lab/shares/DATA/dataset/$1"

if [ ! -f $1 ]; then
	echo  "$1 does not exist"
	exit
fi
touch $dest_file
#!/bin/bash
while IFS='' read -r line || [[ -n "$line" ]]; do
	src_location="$base_path$line"
	echo "Copying files for $line"
	echo "SOURCE LOCATION - $src_location"
	
	#rename the files
	if [ -f "$src_location/outbound.csv" ]; then
		tail -n +2 "$src_location/outbound.csv" >> "$dest_file"
	fi

	if [ -f "$src_location/return.csv" ]; then
		tail -n +2 "$src_location/return.csv" >> "$dest_file"
	fi
done < "$1"