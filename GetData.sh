base_path="/ifs/home/c2b2/ip_lab/shares/DATA/fwd_bwd_data/converted_fb_accel_data/"
dest_path="/ifs/home/c2b2/ip_lab/shares/DATA/dataset/"

if [ ! -f $1 ]; then
	echo  "$1 does not exist"
	exit
fi
#!/bin/bash
while IFS='' read -r line || [[ -n "$line" ]]; do
	src_location="$base_path$line"
	echo "Copying files for $line"
	echo "SOURCE LOCATION - $src_location"
	#remove the .csv extension
	t=$(echo $1 | rev)
	t=${t: 4}
	t=$(echo $t | rev)
	dest_location="$dest_path$t/"
	if [ ! -d "$dest_location" ]; then
		#create the dest location
		mkdir $dest_location
	fi

	#rename the files
	if [ -f "$src_location/outbound.csv" ]; then
		mv "$src_location/outbound.csv" "$src_location/$line-outbound.csv"
		cp "$src_location/$line-outbound.csv" "$dest_location"
		mv "$src_location/$line-outbound.csv" "$src_location/outbound.csv"
	fi

	if [ -f "$src_location/return.csv" ]; then
		mv "$src_location/return.csv" "$src_location/$line-return.csv"
		cp "$src_location/$line-return.csv" "$dest_location"
		mv "$src_location/$line-return.csv" "$src_location/return.csv"
	fi
done < "$1"