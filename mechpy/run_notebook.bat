:: Use to launch jupyter notebooks

:: change console to the current working directory
Pushd "%~dp0"

:: launch jupyter notebook
jupyter notebook

:: write html output
:: jupyter nbconvert --to html mechpy.ipynb

pause
