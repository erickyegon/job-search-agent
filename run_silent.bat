@echo off
cd /d "%~dp0"
for /f "tokens=1,2 delims==" %%a in (.env) do set %%a=%%b
python -m src.main --email --telegram --hours 24 >> logs.txt 2>&1
