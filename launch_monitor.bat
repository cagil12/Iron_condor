@echo off
TITLE Ingresarios Options - Live Monitor 🦅
COLOR 0A
ECHO ========================================================
ECHO    INGRESARIOS OPTIONS RESEARCH - LIVE MONITOR
ECHO ========================================================
ECHO.
ECHO Inicianizando monitoreo de posiciones...
ECHO Hora de inicio: %TIME%
ECHO.

:: Cambiar al directorio del proyecto
cd /d "%~dp0"

:: Ejecutar el script (usando el python del sistema)
python run_live_monitor.py --live

:: Si falla o termina, pausa para ver el error
ECHO.
ECHO ========================================================
ECHO    PROCESO TERMINADO
ECHO ========================================================
PAUSE
