# beagle  bird  duck  girlwalk  horse  torus2sphere
#!/bin/bash

# D-NeRF 数据集中所有场景的列表
SCENES=(
    # "beagle"
    # "bird" 
    # "duck"
    "girlwalk"
    # "horse"
    # "torus2sphere"
)

# 遍历每个场景并训练
for scene in "${SCENES[@]}"
do
    echo "==============================================="
    echo "开始训练场景: $scene"
    echo "==============================================="
    
    # 创建输出目录
    mkdir -p output/exp-lbs-$scene
    
    # 执行训练命令
    python train.py -s ../../../data/dg-mesh/$scene \
                   -m tempexp/exp-lbs-$scene \
                   --is_blender --eval
    
    if [ $? -eq 0 ]; then
        echo "场景 $scene 训练成功完成"
    else
        echo "场景 $scene 训练失败"
    fi
done





# python train_lbs.py -s ../../../data/d-nerf/jumpingjacks -m output/exp-lbs-jumpingjacks --is_blender