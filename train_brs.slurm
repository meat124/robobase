#!/bin/bash

#SBATCH --job-name=train_brs         # 잡 이름 (Job name)
#SBATCH --output=logs/%x_%j.out     # 표준 출력 로그 파일 (%x: job-name, %j: jobid)
#SBATCH --error=logs/%x_%j.err      # 에러 로그 파일
#SBATCH --partition=suma_rtx4090        # 사용할 파티션 이름
#SBATCH --gres=gpu:1                # GPU 1개 요청 (클러스터 설정에 맞게)
#SBATCH --time=48:00:00             # 실행 시간 제한 (HH:MM:SS)

conda activate bigym
echo "Conda environment activated."

cd /scratch2/meat124/bigym_ws/robobase
echo "Current directory is robobase"

echo "BRS policy train started"
srun bash train_brs.sh
echo "BRS policy train ended"