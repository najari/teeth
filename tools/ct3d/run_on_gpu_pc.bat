@echo off
rem Runs the teeth segmentation on a PC with an NVIDIA GPU.
rem Put this file, setup_segmentation.py, segment_teeth.py and ct.nii.gz in one folder, then double-click this file.
setlocal
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Install Python 3.10 or newer from python.org, tick "Add python.exe to PATH", then run this again.
  goto :err
)
if not exist ct.nii.gz (
  echo ct.nii.gz is missing from this folder.
  goto :err
)

echo [1/2] Setting up the environment. The first run downloads several GB.
python setup_segmentation.py "%~dp0.venv-seg"
if errorlevel 1 goto :err

echo [2/2] Running the teeth segmentation.
"%~dp0.venv-seg\Scripts\python.exe" segment_teeth.py ct.nii.gz teeth_seg
if errorlevel 1 goto :err

echo.
echo Done. Copy the "teeth_seg" folder back to the first PC.
pause
exit /b 0

:err
echo.
echo It did not finish. Copy the messages above and send them back.
pause
exit /b 1
