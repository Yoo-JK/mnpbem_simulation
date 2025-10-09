#!/bin/bash

# MNPBEM Automation Pipeline
# Usage: ./master.sh --structure ./config/structures/config_structure.py --simulation ./config/simulations/config_simulation.py

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored message
print_msg() {
    echo -e "${2}${1}${NC}"
}

# Parse command line arguments
STRUCTURE_FILE=""
SIMULATION_FILE=""
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --structure)
            STRUCTURE_FILE="$2"
            shift 2
            ;;
        --simulation)
            SIMULATION_FILE="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help|-h)
            echo "MNPBEM Automation Pipeline"
            echo ""
            echo "Usage: ./master.sh --structure <structure_file> --simulation <simulation_file> [options]"
            echo ""
            echo "Required arguments:"
            echo "  --structure <file>   Structure configuration file path"
            echo "  --simulation <file>  Simulation configuration file path"
            echo ""
            echo "Optional arguments:"
            echo "  --verbose           Enable verbose output"
            echo "  --help, -h          Show this help message"
            echo ""
            echo "Example:"
            echo "  ./master.sh --structure ./config/structures/config_structure.py \\"
            echo "              --simulation ./config/simulations/config_simulation.py"
            exit 0
            ;;
        *)
            print_msg "Error: Unknown option: $1" "$RED"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$STRUCTURE_FILE" ]; then
    print_msg "Error: --structure option is required" "$RED"
    echo "Usage: ./master.sh --structure <structure_file> --simulation <simulation_file>"
    exit 1
fi

if [ -z "$SIMULATION_FILE" ]; then
    print_msg "Error: --simulation option is required" "$RED"
    echo "Usage: ./master.sh --structure <structure_file> --simulation <simulation_file>"
    exit 1
fi

# Validate files exist
if [ ! -f "$STRUCTURE_FILE" ]; then
    print_msg "Error: Structure config file not found: $STRUCTURE_FILE" "$RED"
    exit 1
fi

if [ ! -f "$SIMULATION_FILE" ]; then
    print_msg "Error: Simulation config file not found: $SIMULATION_FILE" "$RED"
    exit 1
fi

# Extract output_dir from simulation config
OUTPUT_DIR=$(python3 -c "
import sys
sys.path.insert(0, '.')
try:
    exec(open('$SIMULATION_FILE').read())
    if 'args' in dir() and 'output_dir' in args:
        print(args['output_dir'])
    else:
        print('./results')
except Exception as e:
    print('./results')
" 2>/dev/null)

if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="./results"
fi

# Start pipeline
print_msg "╔════════════════════════════════════════════════════════════╗" "$BLUE"
print_msg "║         MNPBEM Automation Pipeline Started               ║" "$BLUE"
print_msg "╚════════════════════════════════════════════════════════════╝" "$BLUE"
echo ""
print_msg "📄 Structure config:  $STRUCTURE_FILE" "$BLUE"
print_msg "📄 Simulation config: $SIMULATION_FILE" "$BLUE"
print_msg "📁 Output directory:  $OUTPUT_DIR" "$BLUE"
echo ""

# Create necessary directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/logs"
mkdir -p ./simulation

# Generate timestamp for logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$OUTPUT_DIR/logs"
MATLAB_LOG="$LOG_DIR/matlab_$TIMESTAMP.log"
PYTHON_LOG="$LOG_DIR/pipeline_$TIMESTAMP.log"

# Export environment variables
export MNPBEM_STRUCTURE="$STRUCTURE_FILE"
export MNPBEM_SIMULATION="$SIMULATION_FILE"
export MNPBEM_TIMESTAMP="$TIMESTAMP"
export MNPBEM_LOG_DIR="$LOG_DIR"
export MNPBEM_OUTPUT_DIR="$OUTPUT_DIR"

# Step 1: Generate MATLAB simulation code
print_msg "🔧 Step 1/3: Generating MATLAB simulation code..." "$YELLOW"

# Run simulation generator and capture the run folder
TEMP_OUTPUT=$(mktemp)
if [ "$VERBOSE" = true ]; then
    python run_simulation.py --structure "$STRUCTURE_FILE" --simulation "$SIMULATION_FILE" --verbose 2>&1 | tee "$TEMP_OUTPUT" | tee -a "$PYTHON_LOG"
else
    python run_simulation.py --structure "$STRUCTURE_FILE" --simulation "$SIMULATION_FILE" 2>&1 | tee "$TEMP_OUTPUT" >> "$PYTHON_LOG"
fi

PYTHON_EXIT_CODE=$?
if [ $PYTHON_EXIT_CODE -ne 0 ]; then
    print_msg "✗ Error: Failed to generate MATLAB code" "$RED"
    print_msg "Check log file: $PYTHON_LOG" "$RED"
    rm -f "$TEMP_OUTPUT"
    exit 1
fi

# Extract RUN_FOLDER from output
RUN_FOLDER=$(grep "^RUN_FOLDER=" "$TEMP_OUTPUT" | cut -d'=' -f2)
rm -f "$TEMP_OUTPUT"

if [ -z "$RUN_FOLDER" ]; then
    print_msg "✗ Error: Could not determine run folder" "$RED"
    exit 1
fi

print_msg "✓ MATLAB code generated successfully" "$GREEN"
print_msg "   Run folder: $RUN_FOLDER" "$BLUE"

# Update log paths to use run folder
MATLAB_LOG="$RUN_FOLDER/logs/matlab.log"
PYTHON_LOG="$RUN_FOLDER/logs/pipeline.log"
echo ""

# Extract MNPBEM path from simulation config
print_msg "📂 Reading MNPBEM path from configuration..." "$YELLOW"
MNPBEM_PATH=$(python3 -c "
import sys
sys.path.insert(0, '.')
try:
    exec(open('$SIMULATION_FILE').read())
    if 'args' in dir() and 'mnpbem_path' in args:
        print(args['mnpbem_path'])
    else:
        print('/home/yoojk20/workspace/MNPBEM')
except Exception as e:
    print('/home/yoojk20/workspace/MNPBEM')
" 2>/dev/null)

if [ -z "$MNPBEM_PATH" ]; then
    print_msg "✗ Error: MNPBEM path not found in config" "$RED"
    print_msg "   Please set 'mnpbem_path' in your simulation config file" "$RED"
    exit 1
fi

# Expand ~ to home directory if present
MNPBEM_PATH="${MNPBEM_PATH/#\~/$HOME}"

print_msg "   MNPBEM Path: $MNPBEM_PATH" "$BLUE"

# Verify MNPBEM path exists
if [ ! -d "$MNPBEM_PATH" ]; then
    print_msg "✗ Error: MNPBEM directory not found: $MNPBEM_PATH" "$RED"
    print_msg "   Please check 'mnpbem_path' in your simulation config" "$RED"
    exit 1
fi
print_msg "✓ MNPBEM path verified" "$GREEN"
echo ""

# Step 2: Run MATLAB simulation
print_msg "⚡ Step 2/3: Running MATLAB simulation..." "$YELLOW"

# Check if MATLAB is available
if ! command -v matlab &> /dev/null; then
    print_msg "✗ Error: MATLAB not found in PATH" "$RED"
    exit 1
fi

# Check if simulation script exists
if [ ! -f "./simulation/simulation_script.m" ]; then
    print_msg "✗ Error: simulation_script.m not found" "$RED"
    exit 1
fi

# Run MATLAB with dynamic MNPBEM path
cd simulation
if [ "$VERBOSE" = true ]; then
    matlab -nodisplay -nodesktop -r "addpath(genpath('$MNPBEM_PATH')); run('simulation_script.m'); quit" 2>&1 | tee "../$MATLAB_LOG"
else
    matlab -nodisplay -nodesktop -r "addpath(genpath('$MNPBEM_PATH')); run('simulation_script.m'); quit" > "../$MATLAB_LOG" 2>&1
fi
MATLAB_EXIT_CODE=$?
cd ..

if [ $MATLAB_EXIT_CODE -ne 0 ]; then
    print_msg "✗ Error: MATLAB simulation failed (exit code: $MATLAB_EXIT_CODE)" "$RED"
    print_msg "Check log file: $MATLAB_LOG" "$RED"
    exit 1
fi
print_msg "✓ MATLAB simulation completed successfully" "$GREEN"
echo ""

# Step 3: Postprocess results
print_msg "📊 Step 3/3: Processing and analyzing results..." "$YELLOW"
if [ "$VERBOSE" = true ]; then
    python run_postprocess.py --structure "$STRUCTURE_FILE" --simulation "$SIMULATION_FILE" --verbose 2>&1 | tee -a "$PYTHON_LOG"
else
    python run_postprocess.py --structure "$STRUCTURE_FILE" --simulation "$SIMULATION_FILE" >> "$PYTHON_LOG" 2>&1
fi

if [ $? -ne 0 ]; then
    print_msg "✗ Error: Failed to process results" "$RED"
    print_msg "Check log file: $PYTHON_LOG" "$RED"
    exit 1
fi
print_msg "✓ Results processed successfully" "$GREEN"
echo ""

# Pipeline completed
print_msg "╔════════════════════════════════════════════════════════════╗" "$GREEN"
print_msg "║         Pipeline Completed Successfully! 🎉              ║" "$GREEN"
print_msg "╚════════════════════════════════════════════════════════════╝" "$GREEN"
echo ""
print_msg "📁 Results and logs saved in: $OUTPUT_DIR/" "$BLUE"
print_msg "   ├─ Results: $OUTPUT_DIR/" "$BLUE"
print_msg "   └─ Logs: $OUTPUT_DIR/logs/" "$BLUE"
echo ""
print_msg "Timestamp: $TIMESTAMP" "$BLUE"