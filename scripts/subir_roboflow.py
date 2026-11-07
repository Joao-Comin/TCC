"""
Sobe datasets/novos_unificado/ para o Roboflow (imagens + anotacoes YOLO),
respeitando os splits e marcando cada imagem com tags para priorizar a
revisao manual:

  prio-1           luvas do Grounding DINO (dino-gloves ou talvez-gloves)
  prio-2           oculos do Grounding DINO (dino-goggles ou talvez-goggles)
  prio-3           demais imagens com pre-anotacao do best.pt
  prio-4           imagens so com anotacao original do dataset
  dino-<classe>    ganhou caixa do DINO com score alto (~85-90% certas: conferir)
  talvez-<classe>  DINO viu com score baixo e NAO ganhou caixa (~40%: decidir)
  pre-<classe>     classe que foi pre-anotada pelo best.pt nessa imagem
  origem-<pref>    dataset de origem (epids, ppecv, vestds, autogl, sgogg)
  negativo         imagem sem nenhuma caixa (exemplo negativo)
Limiares do DINO em scripts/aplicar_dino.py.

No Roboflow, filtrar com por exemplo `tag:prio-1` ou `tag:origem-sgogg AND class:helmet`.

Sobe na ordem de prioridade (prio-1 primeiro), entao da para comecar a
revisar antes de terminar. Se cair no meio, rodar de novo: o que ja subiu
fica em upload_log.txt e e pulado.

A chave vai em .env na raiz do projeto (ROBOFLOW_API_KEY=...), nunca no codigo.

Uso:
    venv_win\\Scripts\\python scripts/subir_roboflow.py --dry-run
    venv_win\\Scripts\\python scripts/subir_roboflow.py --limite 5
    venv_win\\Scripts\\python scripts/subir_roboflow.py
"""

import argparse
import csv
import os
import tempfile
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from PIL import Image

WORKSPACE = "tcc-nglng"
DATASET_DIR = "datasets/novos_unificado"
RELATORIO = os.path.join(DATASET_DIR, "relatorio.csv")
DATA_YAML = os.path.join(DATASET_DIR, "data.yaml")
LOG_PATH = os.path.join(DATASET_DIR, "upload_log.txt")


def prioridade(dino, classes_pre):
    if dino["gloves"]:
        return 1
    if dino["goggles"]:
        return 2
    if classes_pre:
        return 3
    return 4


def voc_vazio(img_path, pasta):
    w, h = Image.open(img_path).size
    nome = os.path.basename(img_path)
    xml = os.path.join(pasta, os.path.splitext(nome)[0] + ".xml")
    with open(xml, "w", encoding="utf-8") as f:
        f.write(
            f"<annotation><filename>{nome}</filename><size><width>{w}</width>"
            f"<height>{h}</height><depth>3</depth></size></annotation>"
        )
    return xml


def montar_fila():
    fila = []
    with open(RELATORIO, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            nome = row["arquivo"]
            split = row["split"]
            prefixo = nome.split("_")[0]
            classes_pre = row["classes_pre_anotadas"].split()
            # colunas criadas pelo scripts/aplicar_dino.py (ausentes se ele ainda nao rodou)
            tags_dino = []
            for cls in ("gloves", "goggles"):
                if int(row.get(f"dino_{cls}") or 0):
                    tags_dino.append(f"dino-{cls}")
                if row.get(f"talvez_{cls}") == "1":
                    tags_dino.append(f"talvez-{cls}")
            dino = {cls: any(t.endswith(cls) for t in tags_dino) for cls in ("gloves", "goggles")}
            prio = prioridade(dino, classes_pre)

            img = os.path.join(DATASET_DIR, split, "images", nome)
            lbl = os.path.join(DATASET_DIR, split, "labels", os.path.splitext(nome)[0] + ".txt")

            tags = [f"prio-{prio}", f"origem-{prefixo}"] + [f"pre-{c}" for c in classes_pre] + tags_dino
            negativo = not os.path.exists(lbl) or os.path.getsize(lbl) == 0
            if negativo:
                tags.append("negativo")
            fila.append({
                "nome": nome,
                "split": split,
                "img": img,
                "lbl": lbl,
                "negativo": negativo,
                "prio": prio,
                "tags": tags,
                "batch": "novos-pre-anotadas" if classes_pre or tags_dino else "novos-originais",
                "metadata": {"dataset_origem": row["dataset"], "arquivo_original": row["original"]},
            })
    fila.sort(key=lambda x: x["prio"])
    return fila


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--projeto", default="tcc-epi", help="slug do projeto no workspace")
    ap.add_argument("--dry-run", action="store_true", help="so mostra o que subiria")
    ap.add_argument("--limite", type=int, default=0, help="sobe so N imagens (teste)")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    fila = montar_fila()

    ja_subiu = set()
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, encoding="utf-8") as f:
            ja_subiu = {linha.strip() for linha in f if linha.strip()}
    fila = [x for x in fila if x["nome"] not in ja_subiu]
    if args.limite:
        fila = fila[: args.limite]

    print(f"{len(ja_subiu)} ja enviadas antes; {len(fila)} na fila agora")
    print("por prioridade:", dict(sorted(Counter(x["prio"] for x in fila).items())))
    print("por split:", dict(Counter(x["split"] for x in fila)))
    faltando = [x["nome"] for x in fila if not os.path.exists(x["img"])]
    if faltando:
        print(f"AVISO: {len(faltando)} imagens do relatorio nao existem no disco, ex.: {faltando[:3]}")
    if args.dry_run:
        return

    load_dotenv()
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("ROBOFLOW_API_KEY nao encontrada. Crie o arquivo .env na raiz do projeto.")

    import roboflow

    projeto = roboflow.Roboflow(api_key=api_key).workspace(WORKSPACE).project(args.projeto)

    lock = threading.Lock()
    erros = []

    pasta_voc = tempfile.mkdtemp()

    def enviar(item):
        lbl, labelmap = item["lbl"], DATA_YAML
        if item["negativo"]:
            # .txt YOLO vazio da "Unrecognized annotation format"; um VOC sem
            # <object> e aceito e a imagem entra como exemplo negativo (null)
            lbl, labelmap = voc_vazio(item["img"], pasta_voc), None
        projeto.single_upload(
            image_path=item["img"],
            annotation_path=lbl,
            annotation_labelmap=labelmap,
            split=item["split"],
            batch_name=item["batch"],
            tag_names=item["tags"],
            metadata=item["metadata"],
            num_retry_uploads=3,
            annotation_overwrite=True,  # re-envio substitui a anotacao antiga
        )
        return item

    with open(LOG_PATH, "a", encoding="utf-8") as log, ThreadPoolExecutor(args.threads) as pool:
        futuros = {pool.submit(enviar, x): x for x in fila if os.path.exists(x["img"])}
        for i, fut in enumerate(as_completed(futuros), 1):
            item = futuros[fut]
            try:
                fut.result()
                with lock:
                    log.write(item["nome"] + "\n")
                    log.flush()
            except Exception as e:
                erros.append((item["nome"], str(e)))
            if i % 100 == 0 or i == len(futuros):
                print(f"  {i}/{len(futuros)} (erros: {len(erros)})")

    if erros:
        print(f"\n{len(erros)} falharam (rodar de novo tenta so essas). Primeiras:")
        for nome, e in erros[:10]:
            print(f"  {nome}: {e}")


if __name__ == "__main__":
    main()
