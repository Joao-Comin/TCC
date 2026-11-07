"""
Nucleo da deteccao de EPIs, compartilhado pelo app (app/servidor.py) e
pelos scripts de avaliacao (scripts/avaliar_teste_real.py etc.), para que
os numeros do TCC medam exatamente o pipeline que o app roda.
"""

CLASSES = ["helmet", "vest", "gloves", "boots", "goggles"]
NOMES_PT = {
    "helmet": "Capacete",
    "vest": "Colete",
    "gloves": "Luvas",
    "boots": "Botas",
    "goggles": "Óculos",
}
