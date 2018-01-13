#!/bin/bash
for i in `seq 1 5`;
  do
    echo $i
    ./apps/serial
    ./apps/serial
  done
