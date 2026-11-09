# Detecção de EPI com YOLO — TCC (Unesc)

**Aplicação de Inteligência Artificial na Detecção de Equipamentos de Proteção Individual para Prevenção de Acidentes em Ambientes Metalúrgicos** — João Victor Comin, orientador Sergio Coral.

Um modelo YOLO treinado detecta 5 EPIs (capacete, colete, luvas, botas e óculos). O app web roda o modelo sobre um vídeo ou a webcam, acompanha cada pessoa com um ID e mostra em tempo real quais EPIs ela está usando e quais estão faltando.

## Como rodar

### 1. Pré-requisitos

- **Python 3.12 ou mais novo** ([python.org](https://www.python.org/downloads/); no Windows, marque *"Add Python to PATH"*)
- **Git** (para clonar o repositório)

### 2. Baixar o projeto e instalar as dependências

**Windows**
```bat
git clone https://github.com/Joao-Comin/TCC.git
cd TCC
python -m venv venv_win
venv_win\Scripts\python -m pip install -r requirements.txt
```

**Linux**
```bash
git clone https://github.com/Joao-Comin/TCC.git
cd TCC
python3 -m venv venv
venv/bin/pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```
(o `--extra-index-url` baixa o PyTorch só para CPU, bem menor que a versão com CUDA)

### 3. Criar a pasta de vídeos

O modelo treinado (v5) já vem no repositório, em `models/best.pt`. Só falta criar a pasta onde ficam os vídeos analisados:

```bash
mkdir video
```

### 4. (Opcional) Exportar para OpenVINO — deixa ~2x mais rápido

```bash
venv/bin/python scripts/exportar_openvino.py --modelos v5=models/best.pt
# Windows: venv_win\Scripts\python scripts\exportar_openvino.py --modelos v5=models/best.pt
```

Leva cerca de 2 minutos e só precisa ser feito uma vez. Sem isso o app roda no PyTorch (CPU).

### 5. Iniciar o app

```bash
venv/bin/python app/servidor.py
# Windows: venv_win\Scripts\python app\servidor.py   (ou dois cliques em iniciar_app.bat)
```

Abra **<http://localhost:8000>**. Envie um vídeo (`.mp4`, `.webm`, `.mov`) pela interface ou copie-o para a pasta `video/`; a aba **Câmera ao vivo** usa a webcam.

Histórico das versões do modelo: [CHANGELOG.md](CHANGELOG.md).
