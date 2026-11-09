@echo off
rem Abre o Monitor de EPI (app/servidor.py) e o navegador em http://localhost:8000
rem Para fechar: feche esta janela (ou Ctrl+C).
cd /d "%~dp0"
if not exist models\openvino (
    echo Exportando os modelos para OpenVINO ^(so na primeira vez, ~2 min^)...
    venv_win\Scripts\python scripts\exportar_openvino.py --modelos v5=models/best.pt
)
start "" http://localhost:8000
venv_win\Scripts\python app\servidor.py
pause
