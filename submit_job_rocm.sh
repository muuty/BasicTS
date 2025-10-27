#!/bin/bash

# Get config file from command line argument
CONFIG_FILE=$1

# Extract job name from config file path
# Get the last part of the path and remove .py extension
JOB_NAME=$(basename "$CONFIG_FILE" .py)

echo "Submitting job with name: $JOB_NAME"
echo "Config file: $CONFIG_FILE"

# Create a temporary SBATCH script with the dynamic job name
TEMP_SCRIPT=$(mktemp)
cat > "$TEMP_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --partition=gpu_rocm       
#SBATCH --qos=gpu                
#SBATCH --gres=gpu:mi300x:1              
#SBATCH --mem=32G                 
#SBATCH --time=72:00:00            
#SBATCH --output=logs/%j.log 

# --- 스크립트 시작 ---

echo "Job Name: $JOB_NAME"
echo "Config File: $CONFIG_FILE"
echo "Job Start Time: \$(date)"

eval "\$(conda shell.bash hook)"
conda activate rocm
which python

# 4. 파이썬 스크립트 실행
python experiments/train.py --cfg=$CONFIG_FILE

# 5. 잡 종료 정보 출력
echo "-----------------------------------"
echo "Job End Time: \$(date)"
EOF

# Submit the job using the temporary script
sbatch "$TEMP_SCRIPT"

# Clean up the temporary script
rm "$TEMP_SCRIPT"
