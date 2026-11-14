#!/bin/bash
cd "$(dirname "$0")"
for s in starlink-vleo starlink-480 starlink-550 oneweb-1200; do
  nice -n 10 python3 mc_ta.py --shell $s --n 4000 --seed 7 > out/mc_$s.log 2>&1 &
done
wait
echo done > out/mc.done
