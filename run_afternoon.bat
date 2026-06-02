@echo off
cd /d "%~dp0"
echo Loading environment...
for /f "tokens=1,2 delims==" %%a in (.env) do set %%a=%%b
echo Running Job Search Agent (afternoon)...
python -m src.main --telegram --hours 8
echo Done at %date% %time%
pause
