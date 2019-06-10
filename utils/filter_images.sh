#!/bin/bash

folder=$1
filter_dir="$folder"/filtered_out
mkdir -p "$filter_dir"


#ls -Sl "$folder" | awk '$5 < 122' | cut -d' ' -f 12- | xargs ls

# Very small images
find . -maxdepth 1 -type f -size -1100c -exec mv  \{\} "$filter_dir" \;

# Very large images
find . -maxdepth 1 -type f -size +19000c -exec mv  \{\} "$filter_dir" \;

#find . -maxdepth 1 -type f -exec file  \{\}  \; | awk '{print $(NF-3) " " $(NF-5)}' > "$folder"/res #| sort -nk1
#find . -maxdepth 1 -type f -exec file  \{\}  \; | sed 's/,//g' | awk '{for(i=1; i<=NF; i++) {if($i=="data") print $(i+1),  $(i+2), $(i+3)}}'

# Images with large height
#find . -maxdepth 1 -type f -exec file  \{\}  \; | sed 's/,//g' | awk '{for(i=1; i<=NF; i++) {if($i=="data") if ($(i+3) > 70) print}}'
# That finds only CMU Typewriter Text, let's exclude it entirely

find . -maxdepth 1 -type f -iname "*CMU*" -exec mv  \{\} "$filter_dir" \;

