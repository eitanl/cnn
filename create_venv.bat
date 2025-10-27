@echo off

echo Installing virtual environment
python -m venv env
REM or virtualenv env

echo Activating
call env\Scripts\activate.bat

echo Installing requirements
REM protect us in case that the virtual environment was not set for any reason
set PIP_REQUIRE_VIRTUALENV=true
pip install -r requirements.txt

pip list
