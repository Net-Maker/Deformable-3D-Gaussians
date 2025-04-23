# python train.py -s ../../../data/d-nerf/jumpingjacks -m output/exp-lbs-jumping --is_blender

LIST=("hook" "bouncingballs" "hellwarrior" "mutant" "standup" "trex" "lego" "jumpingjacks")
# LIST2=("beagle" "bird" "duck" "girlwalk" "horse" "torus2sphere")
DATASET="d-nerf"
for ELEMENT in "${LIST2[@]}";do
  echo "run-${ELEMENT}"
  python dgmesh/train.py -s ../../../data/${DATASET}/${ELEMENT} -m output/exp-lbs-${ELEMENT} --is_blender
  done