import os
import pandas as pd
import warnings

try:
    from feat.pipeline import Detector
except ImportError:
    from feat.detector import Detectorv1 as Detector

warnings.filterwarnings("ignore")

print("Carregando Detector com modelos padrão compatíveis...")

# Instanciação sem argumentos manuais conflitantes:
# O Py-Feat seleciona automaticamente o par compatível (RetinaFace + modelo neural de AU/emoção)
detector = Detector()

def extrair_metricas_sutileza(predicoes):
    colunas_au = [col for col in predicoes.columns if col.startswith("AU")]
    colunas_emo = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]
    
    catalogo = []

    for idx, row in predicoes.iterrows():
        arquivo = row.get("input", f"imagem_{idx}")
        
        # 1. Action Units
        aus = row[colunas_au].astype(float)
        au_max_val = aus.max() if not aus.empty else 0.0
        au_max_nome = aus.idxmax() if not aus.empty else "Nenhuma"
        au_media = aus.mean() if not aus.empty else 0.0
        aus_ativas = aus[aus >= 0.20].round(3).to_dict() if not aus.empty else {}

        # 2. Emoções
        emos = row[colunas_emo].astype(float)
        score_neutro = emos.get("neutral", 0.0)
        
        # Remove neutral para encontrar a expressão emocional ativa (seja sutil ou não)
        emos_sem_neutro = emos.drop(labels=["neutral"], errors="ignore")
        emo_expressiva = emos_sem_neutro.idxmax()
        score_expressiva = emos_sem_neutro.max()

        # 3. Classificação Especializada
        if score_neutro >= 0.85 and au_max_val < 0.40:
            tipo_intensidade = "Neutra Basal"
        elif score_neutro >= 0.60 and (0.20 <= au_max_val or score_expressiva >= 0.15):
            tipo_intensidade = "Microexpressão / Vazamento"
        elif score_expressiva >= 0.65 and au_max_val >= 0.50:
            tipo_intensidade = "Alta (Explícita)"
        elif 0.30 <= score_expressiva < 0.65 or (0.25 <= au_max_val < 0.50):
            tipo_intensidade = "Moderada (Sutil)"
        else:
            tipo_intensidade = "Incerta / Difusa"

        catalogo.append({
            "arquivo": os.path.basename(str(arquivo)),
            "emocao_alvo": emo_expressiva,
            "forca_emocao": round(score_expressiva, 3),
            "grau_neutralidade": round(score_neutro, 3),
            "classificacao": tipo_intensidade,
            "au_pico": au_max_nome,
            "forca_au_pico": round(au_max_val, 3),
            "aus_ativas": str(aus_ativas)
        })

    return pd.DataFrame(catalogo)

def processar_catalogo(pasta_imagens):
    extensoes = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
    lista_arquivos = [
        os.path.join(pasta_imagens, f)
        for f in os.listdir(pasta_imagens)
        if f.lower().endswith(extensoes)
    ]
    
    if not lista_arquivos:
        print(f"[!] Nenhuma imagem encontrada em: {pasta_imagens}")
        return

    print(f"Processando {len(lista_arquivos)} imagem(ns)...")
    
    try:
        resultados_brutos = detector.detect(lista_arquivos, batch_size=1)
    except AttributeError:
        resultados_brutos = detector(lista_arquivos, batch_size=1)
    
    df_catalogo = extrair_metricas_sutileza(resultados_brutos)
    
    caminho_csv = os.path.join(pasta_imagens, "catalogo_expressividade.csv")
    df_catalogo.to_csv(caminho_csv, index=False, encoding="utf-8-sig")
    
    print("\n--- Processamento Concluído ---")
    print(f"Catálogo salvo em: {caminho_csv}\n")
    print(df_catalogo[["arquivo", "emocao_alvo", "forca_emocao", "grau_neutralidade", "classificacao", "au_pico", "forca_au_pico"]])

if __name__ == "__main__":
    PASTA_ENTRADA = "./imagens_teste"
    processar_catalogo(PASTA_ENTRADA)