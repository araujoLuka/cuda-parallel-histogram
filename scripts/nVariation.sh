#!/bin/bash

fileName="nVariation.data"
fixedH=2048
fixedNr=20

nStart=10
nEnd=100000000

echo "Running massive executions..."

echo "> fileName=$fileName"
echo "> fixedH=$fixedH"
echo "> fixedNr=$fixedNr"

echo "> nStart=$nStart"
echo "> nEnd=$nEnd"

echo "---"

cd ../
mkdir -p results
rm -f $fileName
touch $fileName

for (( n=$nStart; n<=$nEnd; ))
do
    echo "Running for $n"
    echo "n=$n h=$fixedH nr=$fixedNr" | tee -a "$fileName"
    make nv-script ARGS_N=$n ARGS_H=$fixedH ARGS_NR=$fixedNr | tee -a $fileName
    echo "---" | tee -a $fileName

    if [ $n -lt 100 ]
    then
        n=$((n+10))
    elif [ $n -lt 300 ]
    then
        n=$((n+20))
    elif [ $n -lt 500 ]
    then
        n=$((n+50))
    elif [ $n -lt 1000 ]
    then
        n=$((n+100))
    elif [ $n -eq 1000 ]
    then
        n=$((n+24))
    elif [ $n -eq 1024 ]
    then
        sleep 1
        n=$((n+76))
    elif [ $n -lt 2000 ]
    then
        n=$((n+100))
    elif [ $n -lt 5000 ]
    then
        n=$((n+250))
    elif [ $n -lt 10000 ]
    then
        n=$((n+500))
    elif [ $n -lt 50000 ]
    then
        n=$((n+1000))
    elif [ $n -lt 100000 ]
    then
        n=$((n+2000))
    elif [ $n -lt 1000000 ]
    then
        n=$((n+100000))
    elif [ $n -lt 2000000 ]
    then
        n=$((n+500000))
    elif [ $n -lt 5000000 ]
    then
        n=$((n+1000000))
    elif [ $n -lt 10000000 ]
    then
        n=$((n+2500000))
    elif [ $n -lt 100000000 ]
        then
        n=$((n+5000000))
    else
        n=$((n+100000000))
    fi
done

# remove \r from file
sed -i 's/\r//g' $fileName
mv $fileName results/
