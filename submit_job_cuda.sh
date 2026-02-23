#!/bin/bash

# Global counter for total submitted jobs
TOTAL_SUBMITTED=0

# Function to submit jobs for a single config file
submit_config_file() {
    local CONFIG_FILE=$1
    local TOTAL_RUN=$2
    
    # Extract base job name from config file path
    # Get the last part of the path and remove .py extension
    local BASE_JOB_NAME=$(basename "$CONFIG_FILE" .py)
    
    echo "Submitting jobs for config: $CONFIG_FILE"
    echo "Total runs: $TOTAL_RUN"
    
    # Loop through each run from 1 to TOTAL_RUN
    for RUN in $(seq 1 $TOTAL_RUN); do
        # Create job name with run number
        local JOB_NAME="${BASE_JOB_NAME}_run${RUN}"
        
        echo "Submitting job: $JOB_NAME (run=$RUN)"
        
        # Create a temporary SBATCH script with the dynamic job name
        local TEMP_SCRIPT=$(mktemp)
        cat > "$TEMP_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu                
#SBATCH --gres=gpu:1              
#SBATCH --mem=32G                 
#SBATCH --time=72:00:00            
#SBATCH --output=logs/%j.log 

# --- 스크립트 시작 ---

echo "Job Name: $JOB_NAME"
echo "Config File: $CONFIG_FILE"
echo "Run: $RUN"
echo "Job Start Time: \$(date)"

eval "\$(conda shell.bash hook)"
conda activate cuda
which python

# 4. 파이썬 스크립트 실행 (--run 인자 포함)
python experiments/train.py --cfg=$CONFIG_FILE --run=$RUN

# 5. 잡 종료 정보 출력
echo "-----------------------------------"
echo "Job End Time: \$(date)"
EOF

        # Submit the job using the temporary script
        if sbatch "$TEMP_SCRIPT" > /dev/null 2>&1; then
            TOTAL_SUBMITTED=$((TOTAL_SUBMITTED + 1))
            echo "Submitted: $JOB_NAME"
        else
            echo "Failed to submit: $JOB_NAME"
        fi
        
        # Clean up the temporary script
        rm "$TEMP_SCRIPT"
    done
}

# Get input path and total_run from command line arguments
INPUT_PATH=$1
TOTAL_RUN=$2

# Check if input is a directory or a file
if [ -d "$INPUT_PATH" ]; then
    # Input is a directory - process all .py files in it
    echo "Input is a directory: $INPUT_PATH"
    echo "Total runs per config: $TOTAL_RUN"
    echo "========================================"
    
    # Find all .py files in the directory
    CONFIG_FILES=$(find "$INPUT_PATH" -maxdepth 1 -name "*.py" -type f | sort)
    
    if [ -z "$CONFIG_FILES" ]; then
        echo "No .py files found in directory: $INPUT_PATH"
        exit 1
    fi
    
    # Count config files
    CONFIG_COUNT=$(echo "$CONFIG_FILES" | wc -l)
    echo "Found $CONFIG_COUNT config file(s)"
    echo ""
    
    # Process each config file
    CONFIG_INDEX=0
    while IFS= read -r CONFIG_FILE; do
        CONFIG_INDEX=$((CONFIG_INDEX + 1))
        echo "[$CONFIG_INDEX/$CONFIG_COUNT] Processing: $CONFIG_FILE"
        submit_config_file "$CONFIG_FILE" "$TOTAL_RUN"
        echo ""
    done <<< "$CONFIG_FILES"
    
    echo "========================================"
    echo "Total jobs submitted: $TOTAL_SUBMITTED"
    
elif [ -f "$INPUT_PATH" ]; then
    # Input is a file - process it directly
    echo "Input is a file: $INPUT_PATH"
    submit_config_file "$INPUT_PATH" "$TOTAL_RUN"
    echo "Total jobs submitted: $TOTAL_SUBMITTED"
else
    echo "Error: $INPUT_PATH is neither a file nor a directory"
    exit 1
fi
