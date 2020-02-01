#!/bin/bash

echo 'Starting Jupyter server'
nohup jupyter notebook --port 8888 --ip=0.0.0.0 --allow-root >> notebook/jupyter.log 2>&1 &

echo 'Hit below on local to establish port forwarding connection to server:'
echo '    $ gcloud compute ssh HOST -- -N -L 8888:localhost:8888'

# Wait $MAX seconds until the server starts
# Command below will show a progress bar
MAX=15
COUNT=0
while [ $COUNT -le $MAX ]
do
    BAR=""
    for i in `seq 1 $MAX`
    do
	if [ $COUNT -eq $i ]
	then
	    BAR="$BAR>"
	else
	    if [ $i -le $COUNT ]
	    then
	        BAR="$BAR="
	    else
	        BAR="$BAR "
	    fi
	fi
    done
    PERCENTAGE=`expr $COUNT \* 100 / $MAX`
    PERCENTAGE=`printf %3d $PERCENTAGE`
    WAIT=`expr $MAX - $COUNT`
    WAIT=`printf %2d $WAIT`
    echo -en "Starting Jupyter server. Wait $WAIT s...  [$BAR] $PERCENTAGE %\r"
    sleep 1
    COUNT=`expr $COUNT + 1`
done
echo -e "Starting Jupyter server. Wait $WAIT s...  [$BAR] $PERCENTAGE %\r"

echo 'Started. URL for Jupyter is:'
tail -n 1 notebook/jupyter.log | awk '{print substr($0, index($0, "http"))}' | xargs -n1 echo '    '
