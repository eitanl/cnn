#!/bin/bash

echo "Installing virtual environment"
python -m venv env
# or virtualenv env

echo "Activating"
source env/Scripts/activate

echo "Installing requirements"
pip install -r requirements.txt

echo
pip list

echo
echo "Now run 'source env/Scripts/activate' to activate the virtual environment"
# in Linux the activation is lost once the script exists (because it's done in a subshell)