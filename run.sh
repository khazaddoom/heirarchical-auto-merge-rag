#!/bin/bash

# Configuration
VENV_DIR=".venv"
REQUIREMENTS_FILE="requirements.txt"
ENV_FILE=".env"
ENV_EXAMPLE=".env.example"
MAIN_SCRIPT="main.py"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}>>> Starting Setup and Run Script...${NC}"

# 1. Provide .env if it does not exist
if [ ! -f "$ENV_FILE" ]; then
    if [ -f "$ENV_EXAMPLE" ]; then
        echo -e "${YELLOW}>>> Creating $ENV_FILE from $ENV_EXAMPLE.${NC}"
        cp "$ENV_EXAMPLE" "$ENV_FILE"
        echo -e "${RED}>>> IMPORTANT: Please update $ENV_FILE with your actual API keys before running the app properly.${NC}"
        # We continue even if the user hasn't edited it, because main.py will handle the missing key print out gracefully
    else
        echo -e "${RED}>>> Warning: Neither $ENV_FILE nor $ENV_EXAMPLE found.${NC}"
    fi
else
    echo -e "${GREEN}>>> $ENV_FILE already exists.${NC}"
fi

# 2. Check for Python installation
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    echo -e "${RED}>>> Error: Python is not installed. Please install Python 3.8+ to proceed.${NC}"
    exit 1
fi

# 3. Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}>>> Creating virtual environment in $VENV_DIR...${NC}"
    $PYTHON_CMD -m venv "$VENV_DIR"
else
    echo -e "${GREEN}>>> Virtual environment $VENV_DIR already exists.${NC}"
fi

# 4. Activate virtual environment
echo -e "${GREEN}>>> Activating virtual environment...${NC}"
source "$VENV_DIR/bin/activate"

# 5. Install/Update dependencies
if [ -f "$REQUIREMENTS_FILE" ]; then
    echo -e "${GREEN}>>> Installing dependencies from $REQUIREMENTS_FILE...${NC}"
    pip install --upgrade pip
    pip install -r "$REQUIREMENTS_FILE"
else
    echo -e "${RED}>>> Error: $REQUIREMENTS_FILE not found.${NC}"
    exit 1
fi

# 6. Run the application
if [ -f "$MAIN_SCRIPT" ]; then
    echo -e "${GREEN}>>> Starting the application ($MAIN_SCRIPT)...${NC}"
    echo "--------------------------------------------------------"
    python "$MAIN_SCRIPT"
else
    echo -e "${RED}>>> Error: $MAIN_SCRIPT not found.${NC}"
    exit 1
fi
