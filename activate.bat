@echo off
echo Activating Entity Extraction environment...
call %~dp0venv\Scripts\activate.bat
echo.
echo Environment ready! You can now run:
echo   - .\run.bat                     (Run the main application)
echo   - python src\main.py            (Run the main script directly)
echo   - jupyter notebook              (Start Jupyter notebook server)
echo   - jupyter notebook notebooks\ner_training_example.ipynb (Open example notebook)
echo.
set PYTHONPATH=%~dp0;%PYTHONPATH%
