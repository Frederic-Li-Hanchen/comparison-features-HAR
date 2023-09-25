#!/bin/bash

start=$SECONDS

for idx in $(seq 1 30);
do
	python trainDnn.py $idx; python extractDnnFeatures.py $idx; python svmClassify.py $idx
done

duration=($SECONDS - $start)

echo $duration