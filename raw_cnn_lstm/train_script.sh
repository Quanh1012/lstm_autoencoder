#!/usr/bin/bash

CONFIG_DIR="configs"

for STR in $CONFIG_DIR/*.yml; do
	substring=true;
	for SUB in "$@"; do
		if [[ "$STR" != *"$SUB"* ]]; then
			substring=false;
		fi;
	done;
	if [ "$substring" == true ]; then
		echo "Training with config $STR";
		time python train.py --config $STR;
		if [ $? -eq 0 ]; then
			echo "Testing with config $STR";
			time python inference.py --config $STR;
			if [ $? -eq 0 ]; then
				rm $STR ;
			fi;
            printf "\n";
			printf '=%.0s' {1..100};
            printf "\n";
		fi;
	fi;
done
# RESULTS_DIR="results/buzz2"
# for STR in $RESULTS_DIR/*.yml; do
# 	substring=true;
# 	for SUB in "$@"; do
# 		if [[ "$STR" != *"$SUB"* ]]; then
# 			substring=false;
# 		fi;
# 	done;
# 	if [ "$substring" == true ]; then
# 		echo "Testing with config $STR";
# 		python inference.py --config $STR;
# 	fi;
# done