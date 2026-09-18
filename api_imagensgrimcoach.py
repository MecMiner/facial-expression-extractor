import os
import shutil
import tempfile
import uuid
import warnings
import cv2
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

try:
    from feat.pipeline import Detector
except ImportError:
    from feat.detector import Detectorv1 as Detector

warnings.filterwarnings("ignore")

# Caminhos base
DIR_PROJETO = os.path.dirname(os.path.abspath(__file__))
DIR_SAIDA_DATASET = os.path.join(DIR_PROJETO, "dataset_extraido")
os.makedirs(DIR_SAIDA_DATASET, exist_ok=True)

# Configuração FACS e Níveis
FAIXAS_NIVEIS = {
    "Nivel_0_Basal": (0.00, 0.15),
    "Nivel_1_Micro": (0.15, 0.35),
    "Nivel_2_Sutil": (0.35, 0.58),
    "Nivel_3_Moderada": (0.58, 0.78),
    "Nivel_4_Apice": (0.78, 1.01)
}

MAPA_EMOCOES_PT = {
    "anger": "raiva",
    "disgust": "nojo",
    "fear": "medo",
    "happiness": "alegria",
    "sadness": "tristeza",
    "surprise": "surpresa",
    "neutral": "neutro"
}

AUS_EXPRESSIVAS = [
    "AU01", "AU02", "AU04", "AU05", "AU06", "AU07",
    "AU09", "AU10", "AU12", "AU14", "AU15", "AU17",
    "AU20", "AU23", "AU24", "AU25", "AU26"
]

# Inicialização global do detector
print("[+] Carregando modelos do Py-Feat na memória...")
detector = Detector()
print("[OK] Detector pronto para requisições.")

# Criação da aplicação FastAPI
app = FastAPI(
    title="API de Catalogação Facial - GrimCoach",
    description="Endpoint para ingestão de vídeos e extração de faces padronizadas por nível e emoção."
)

# Mapeia a pasta de dataset para servir as imagens via HTTP
app.mount("/static/dataset", StaticFiles(directory=DIR_SAIDA_DATASET), name="dataset")

def recortar_e_padronizar_rosto(frame_bgr, bbox, tamanho_padrao=512, padding_ratio=0.42):
    h_img, w_img, _ = frame_bgr.shape
    x, y, w, h = bbox

    if w <= 0 or h <= 0 or np.isnan(x) or np.isnan(y):
        menor = min(h_img, w_img)
        cx, cy = w_img // 2, h_img // 2
        recorte = frame_bgr[cy - menor//2 : cy + menor//2, cx - menor//2 : cx + menor//2]
        return cv2.resize(recorte, (tamanho_padrao, tamanho_padrao), interpolation=cv2.INTER_AREA)

    centro_x = x + w / 2.0
    centro_y = y + h / 2.0
    lado = max(w, h) * (1.0 + padding_ratio)

    x1 = int(round(centro_x - lado / 2.0))
    y1 = int(round(centro_y - lado / 2.0))
    x2 = int(round(centro_x + lado / 2.0))
    y2 = int(round(centro_y + lado / 2.0))

    pad_t = max(0, -y1)
    pad_e = max(0, -x1)
    pad_b = max(0, y2 - h_img)
    pad_d = max(0, x2 - w_img)

    recorte = frame_bgr[max(0, y1):min(h_img, y2), max(0, x1):min(w_img, x2)]

    if pad_t > 0 or pad_b > 0 or pad_e > 0 or pad_d > 0:
        recorte = cv2.copyMakeBorder(recorte, pad_t, pad_b, pad_e, pad_d, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return cv2.resize(recorte, (tamanho_padrao, tamanho_padrao), interpolation=cv2.INTER_AREA)

def extrair_frames_temporarios(caminho_video, skip_frames=3):
    cap = cv2.VideoCapture(caminho_video)
    pasta_temp = tempfile.mkdtemp(prefix="pyfeat_api_")
    caminhos = []
    indices = []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx % skip_frames == 0:
            caminho_img = os.path.join(pasta_temp, f"f_{idx:06d}.png")
            cv2.imwrite(caminho_img, frame)
            caminhos.append(caminho_img)
            indices.append(idx)
        idx += 1

    cap.release()
    return pasta_temp, caminhos, indices, total_frames

def processar_com_calibracao_baseline(predicoes, indices_frames, prefixo_origem=""):
    colunas_emo = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]
    colunas_au = [col for col in predicoes.columns if col in AUS_EXPRESSIVAS]

    n_baseline = min(3, len(predicoes))
    baseline_aus = predicoes[colunas_au].iloc[:n_baseline].astype(float).median()

    catalogo = []

    for idx, row in predicoes.iterrows():
        f_real = indices_frames[idx] if idx < len(indices_frames) else idx

        bx, by = float(row.get("FaceRectX", 0.0)), float(row.get("FaceRectY", 0.0))
        bw, bh = float(row.get("FaceRectWidth", 0.0)), float(row.get("FaceRectHeight", 0.0))

        emos = row[colunas_emo].astype(float)
        score_neutro = emos.get("neutral", 0.0)
        emos_sem_neutro = emos.drop(labels=["neutral"], errors="ignore")
        emo_ingles = emos_sem_neutro.idxmax() if not emos_sem_neutro.empty else "neutral"
        score_emocao = emos_sem_neutro.max() if not emos_sem_neutro.empty else 0.0

        aus_raw = row[colunas_au].astype(float)
        aus_delta = (aus_raw - baseline_aus).clip(lower=0.0)
        
        top3_aus = aus_delta.nlargest(3)
        au_pico_val = top3_aus.iloc[0] if not top3_aus.empty else 0.0
        au_pico_nome = top3_aus.index[0] if not top3_aus.empty else "Nenhuma"
        volume_motor = (top3_aus.iloc[0] * 0.6) + (top3_aus.iloc[1] * 0.3) + (top3_aus.iloc[2] * 0.1) if len(top3_aus) >= 3 else au_pico_val

        if score_neutro >= 0.85 and volume_motor < 0.15:
            forca_final = 0.05
            classificacao = "Nivel_0_Basal"
            emo_pt = "neutro"
        else:
            emo_pt = MAPA_EMOCOES_PT.get(emo_ingles, emo_ingles)
            forca_final = (volume_motor * 0.75) + (score_emocao * 0.25)
            forca_final = float(np.clip(forca_final, 0.0, 1.0))

            if forca_final < 0.15:
                classificacao = "Nivel_0_Basal"
            elif 0.15 <= forca_final < 0.35:
                classificacao = "Nivel_1_Micro"
            elif 0.35 <= forca_final < 0.58:
                classificacao = "Nivel_2_Sutil"
            elif 0.58 <= forca_final < 0.78:
                classificacao = "Nivel_3_Moderada"
            else:
                classificacao = "Nivel_4_Apice"

        catalogo.append({
            "arquivo": f"{prefixo_origem}_f{f_real:05d}",
            "frame": int(f_real),
            "emocao_alvo": emo_pt,
            "confianca_emocao": round(score_emocao, 3),
            "grau_neutralidade": round(score_neutro, 3),
            "classificacao": classificacao,
            "au_pico": au_pico_nome,
            "delta_au_pico": round(au_pico_val, 3),
            "forca_final": round(forca_final, 3),
            "bbox_x": bx,
            "bbox_y": by,
            "bbox_w": bw,
            "bbox_h": bh
        })

    return pd.DataFrame(catalogo)

def processar_e_salvar_niveis(caminho_video, df_timeline, base_url, max_frames=3, tamanho_padrao=512):
    identificador = f"rec_{uuid.uuid4().hex[:8]}"
    cap = cv2.VideoCapture(caminho_video)
    resultados = []

    emocoes_presentes = df_timeline["emocao_alvo"].unique()

    for nivel, (min_val, max_val) in FAIXAS_NIVEIS.items():
        candidatos_nivel = df_timeline[(df_timeline["forca_final"] >= min_val) & (df_timeline["forca_final"] < max_val)].copy()
        
        if candidatos_nivel.empty:
            continue

        for emo in emocoes_presentes:
            candidatos = candidatos_nivel[candidatos_nivel["emocao_alvo"] == emo].copy()
            if candidatos.empty:
                continue

            # Pasta física: dataset_extraido / <Nivel> / <Emocao>
            pasta_destino = os.path.join(DIR_SAIDA_DATASET, nivel, emo)
            os.makedirs(pasta_destino, exist_ok=True)

            ponto_medio = (min_val + max_val) / 2
            candidatos["dist_centro"] = (candidatos["forca_final"] - ponto_medio).abs()
            selecionados = candidatos.sort_values(by="dist_centro").head(max_frames)

            for _, row in selecionados.iterrows():
                f_num = int(row["frame"])
                forca = row["forca_final"]
                au = row["au_pico"]
                bbox = (row["bbox_x"], row["bbox_y"], row["bbox_w"], row["bbox_h"])

                cap.set(cv2.CAP_PROP_POS_FRAMES, f_num)
                sucesso, frame_bruto = cap.read()
                if sucesso:
                    rosto = recortar_e_padronizar_rosto(
                        frame_bruto, bbox, tamanho_padrao=tamanho_padrao, padding_ratio=0.42
                    )
                    nome_arquivo = f"{identificador}_f{f_num:05d}_{au}_{forca:.2f}.png"
                    caminho_disco = os.path.join(pasta_destino, nome_arquivo)
                    cv2.imwrite(caminho_disco, rosto)

                    # Constrói URL pública para o banco de dados
                    url_imagem = f"{base_url}static/dataset/{nivel}/{emo}/{nome_arquivo}"

                    resultados.append({
                        "nivel": nivel,
                        "emocao": emo,
                        "forca": forca,
                        "au_pico": au,
                        "url": url_imagem,
                        "caminho_local": caminho_disco
                    })

    cap.release()
    return resultados

# Endpoint Principal de Processamento
@app.post("/api/v1/enviar-video")
async def receber_video(
    request: Request,
    video: UploadFile = File(...),
    skip_frames: int = Form(3),
    amostras_por_nivel: int = Form(2),
    tamanho_imagem: int = Form(512)
):
    # 1. Validação da extensão
    ext = os.path.splitext(video.filename)[1].lower()
    if ext not in [".mp4", ".mov", ".avi", ".mkv", ".webm"]:
        raise HTTPException(status_code=400, detail="Formato de vídeo não suportado. Envie .mp4, .mov, .avi ou .webm.")

    temp_video_path = None
    pasta_frames_temp = None

    try:
        # 2. Salva o arquivo enviado em um arquivo temporário
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_vid:
            shutil.copyfileobj(video.file, temp_vid)
            temp_video_path = temp_vid.name

        # 3. Decodificação dos quadros com OpenCV
        pasta_frames_temp, lista_arquivos, indices_frames, total_frames = extrair_frames_temporarios(
            temp_video_path, skip_frames=skip_frames
        )

        if not lista_arquivos:
            raise HTTPException(status_code=422, detail="Não foi possível decodificar quadros do vídeo enviado.")

        # 4. Inferência com Py-Feat
        try:
            res_bruto = detector.detect(lista_arquivos)
        except AttributeError:
            res_bruto = detector(lista_arquivos)

        # 5. Calibração fisiológica de baseline
        df_calibrado = processar_com_calibracao_baseline(
            res_bruto, indices_frames, prefixo_origem="web_upload"
        )

        # 6. Recorte, padronização e estruturação por Nível / Emoção
        base_url = str(request.base_url)
        imagens_geradas = processar_e_salvar_niveis(
            temp_video_path, df_calibrado, base_url=base_url,
            max_frames=amostras_por_nivel, tamanho_padrao=tamanho_imagem
        )

        return JSONResponse(
            status_code=200,
            content={
                "status": "sucesso",
                "mensagem": "Vídeo processado e faces catalogadas com sucesso.",
                "total_imagens_geradas": len(imagens_geradas),
                "imagens": imagens_geradas
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno no processamento: {str(e)}")

    finally:
        # Limpeza de arquivos temporários de processamento
        if temp_video_path and os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
            except OSError:
                pass
        if pasta_frames_temp and os.path.exists(pasta_frames_temp):
            shutil.rmtree(pasta_frames_temp, ignore_errors=True)

if __name__ == "__main__":
    import uvicorn
    # Executa o servidor na porta 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)