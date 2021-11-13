#!/bin/bash

echo Number of jobs to execute

read num_jobs

for i in $(seq 1 $num_jobs)
do
  echo Executing job $i
  bsub python runner.py do_it
  sleep 10
done


