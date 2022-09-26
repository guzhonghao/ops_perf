echo $1
echo $2
ncu --print-summary per-kernel  --section SpeedOfLight python $1 $2
