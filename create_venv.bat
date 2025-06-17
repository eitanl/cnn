@echo off
echo Note: Run this script from Window's Command Prompt (not PowerShell)
REM Window's PowerShell has its own activation script (Activate.ps1) and its own way calling it
pause

echo Installing virtual environment
python -m venv env
REM or virtualenv env

echo Activating
call env\Scripts\activate.bat

echo Installing requirements
pip install -r requirements.txt

pip list