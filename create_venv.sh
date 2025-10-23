#!/bin/bash

echo "Installing virtual environment"
python -m venv env
# or virtualenv env

echo "Activating"
source env/Scripts/activate

echo "Installing requirements"
export PIP_REQUIRE_VIRTUALENV=true
# PIP_REQUIRE_VIRTUALENV=true protects us in case that the virtual environment was not set for any reason
pip install -r requirements.txt

echo
pip list

echo
echo "Now run 'source env/Scripts/activate' to activate the virtual environment"
# in Linux the activation is lost once the script exists (because it's done in a subshell)