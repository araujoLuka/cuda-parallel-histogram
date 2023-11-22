#!/bin/bash

fileName="hVariation.data"
fixedN=100000000
fixedNr=20

hStart=2
# end based on shared memory size (48KB) per int (4B)
hEnd=12288

echo "Running massive executions..."

echo "> fileName=$fileName"
echo "> fixedN=$fixedN"
echo "> fixedNr=$fixedNr"
echo ""
echo "> hStart=$hStart"
echo "> hEnd=$hEnd"

echo "---"

cd ../
mkdir -p results
rm -f $fileName
touch $fileName

for (( h=$hStart; h<=$hEnd; ))
do
    echo "Running for $n"
    echo "n=$fixedN h=$h nr=$fixedNr" | tee -a "$fileName"
    make nv-script ARGS_N=$fixedN ARGS_H=$h ARGS_NR=$fixedNr ARGS_EXTRA="--no-serial" | tee -a $fileName
    echo "---" | tee -a $fileName

    if [ $h -lt 1024 ]
    then
        h=$((h*2))
    else
        h=$((h+512))
    fi
done

# remove \r from file
sed -i 's/\r//g' $fileName
mv $fileName results/
