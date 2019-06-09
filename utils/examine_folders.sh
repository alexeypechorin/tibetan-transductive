#!/bin/bash

function analyze_files {
    base_path=$1
    shift
    files="$@"

    echo $base_path
    for filename in $files; do
        echo $filename;
        full_file_path=${base_path}${filename};

        du -h --max-depth=0 $full_file_path | awk '{print $1}';

        if [ -f "$full_file_path" ]; then
        #    echo "$full_file_path file"
            echo $(wc -l "$full_file_path" | awk '{print $1}') " lines in file"
        else
        #    echo "$full_file_path dir"
            echo $(find "$full_file_path" | wc -l | awk '{print $1}') " files in directory"
        fi
        echo
    done;
}


wiener_base_path=/specific/disk1/home/alexeyp/Tibetan_Transductive/Data/wiener/
wiener_files='Synthetic/PreparedFullEqualized/data_train.txt Synthetic/PreparedFullEqualized/Images  Original/Prepared20Images/im2line.txt  Original/Prepared20Images/LineImages  Synthetic/PreparedFullEqualized/data_val.txt Synthetic/PreparedFullEqualized/Images char_to_class.pkl'


tibetan_base_path=/specific/disk1/home/alexeyp/Tibetan_Transductive/Data/
tibetan_files='Synthetic/Prepared/data_train.txt  Synthetic/Prepared/Images  Original/Prepared/im2line.txt Original/Prepared/LineImages Synthetic/Prepared/data_val.txt  Synthetic/Prepared/Images  char_to_class.pkl'


analyze_files $wiener_base_path $wiener_files > wiener_result
analyze_files $tibetan_base_path $tibetan_files > tibetan_result

paste wiener_result tibetan_result