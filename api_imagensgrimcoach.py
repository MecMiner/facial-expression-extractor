import os
import shutil
import tempfile
import uuid
import hashlib
import secrets
import smtplib
import warnings
import logging
import traceback
import asyncio
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import cv2
import numpy as np
import pandas as pd
from pymongo import MongoClient

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# ==========================================
# CONFIGURAÇÃO DE LOGS
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("GrimCoachWorker")

try:
    from feat.pipeline import Detector
except ImportError:
    from feat.detector import Detectorv1 as Detector

warnings.filterwarnings("ignore")

# ==========================================
# ESTRUTURA DE DIRETÓRIOS PLANOS (STORAGE)
# ==========================================
DIR_PROJETO = os.path.dirname(os.path.abspath(__file__))
DIR_STORAGE = os.path.join(DIR_PROJETO, "storage")
DIR_STORAGE_VIDEOS = os.path.join(DIR_STORAGE, "videos")
DIR_STORAGE_FACES = os.path.join(DIR_STORAGE, "faces")
DIR_VIDEOS_FILA = os.path.join(DIR_STORAGE, "fila_processamento")

# Criação defensiva de todos os diretórios no início
for pasta in [DIR_STORAGE, DIR_STORAGE_VIDEOS, DIR_STORAGE_FACES, DIR_VIDEOS_FILA]:
    os.makedirs(pasta, exist_ok=True)

# ==========================================
# BANCO DE DADOS (MONGODB)
# ==========================================
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
try:
    mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    db = mongo_client["grimcoach_db"]
    colecao_envios = db["contribuicoes_faciais"]
    logger.info("Conectado ao MongoDB com sucesso.")
except Exception as e:
    logger.error(f"Erro ao conectar com o MongoDB: {e}")

# Controle de concorrência: 1 vídeo processado por vez no hardware local
SEMAFORO_EXECUCAO = asyncio.Semaphore(1)

# ==========================================
# CONFIGURAÇÃO SMTP (ENVIO DE E-MAIL)
# ==========================================
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_FROM_NAME = "GrimCoach - Coleta Facial"

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
    "neutral": "neutro",
    "contempt": "desprezo"
}

EMOCOES_PERMITIDAS = ["alegria", "tristeza", "medo", "raiva", "nojo", "surpresa", "desprezo"]

AUS_EXPRESSIVAS = [
    "AU01", "AU02", "AU04", "AU05", "AU06", "AU07",
    "AU09", "AU10", "AU12", "AU14", "AU15", "AU17",
    "AU20", "AU23", "AU24", "AU25", "AU26"
]

logger.info("Carregando modelos do Py-Feat na memória local...")
detector = Detector()
logger.info("Detector Py-Feat pronto para execuções.")

# ==========================================
# FASTAPI SETUP
# ==========================================
app = FastAPI(
    title="GrimCoach - Worker de Processamento Facial",
    description="Worker local dedicado à ingestão, extração e persistência estruturada de dados faciais."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servidores estáticos apontados para as pastas planas
app.mount("/static/faces", StaticFiles(directory=DIR_STORAGE_FACES), name="faces")
app.mount("/static/videos", StaticFiles(directory=DIR_STORAGE_VIDEOS), name="videos")

# ==========================================
# DISPARO DE E-MAIL
# ==========================================
def disparar_email_smtp(destinatario: str, assunto: str, corpo_html: str):
    if not SMTP_USER or not SMTP_PASSWORD:
        logger.warning(f"[EMAIL SIMULADO] Para: {destinatario} | Assunto: {assunto}")
        return

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = assunto
        msg["From"] = f"{SMTP_FROM_NAME} <{SMTP_USER}>"
        msg["To"] = destinatario
        msg.attach(MIMEText(corpo_html, "html"))

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_USER, destinatario, msg.as_string())
        
        logger.info(f"[EMAIL] Notificação enviada para: {destinatario}")
    except Exception as e:
        logger.error(f"[EMAIL] Erro ao enviar para {destinatario}: {str(e)}")

# ==========================================
# PROCESSAMENTO VISUAL E ARQUIVOS
# ==========================================
def calcular_hash_sha256(caminho_arquivo: str) -> str:
    hasher = hashlib.sha256()
    with open(caminho_arquivo, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

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

def extrair_frames_temporarios(caminho_video, skip_frames=2):
    cap = cv2.VideoCapture(caminho_video)
    pasta_temp = tempfile.mkdtemp(prefix="pyfeat_tmp_")
    caminhos = []
    mapa_frames = {}
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx % skip_frames == 0:
            nome_arquivo = f"f_{idx:06d}.png"
            caminho_img = os.path.join(pasta_temp, nome_arquivo)
            cv2.imwrite(caminho_img, frame)
            caminhos.append(caminho_img)
            mapa_frames[nome_arquivo] = {
                "frame": int(idx),
                "timestamp": float(round(idx / fps, 3))
            }
        idx += 1

    cap.release()
    return pasta_temp, caminhos, mapa_frames, fps

def analisar_quadros_calibrados(predicoes, mapa_frames):
    colunas_emo = [col for col in ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral", "contempt"] if col in predicoes.columns]
    colunas_au = [col for col in predicoes.columns if col in AUS_EXPRESSIVAS]

    n_baseline = min(3, len(predicoes))
    baseline_aus = predicoes[colunas_au].iloc[:n_baseline].astype(float).median() if not predicoes.empty else pd.Series(0, index=colunas_au)

    catalogo = []
    for row_idx in range(len(predicoes)):
        row = predicoes.iloc[row_idx]
        
        nome_img_pred = ""
        if "input" in row and pd.notna(row["input"]):
            nome_img_pred = os.path.basename(str(row["input"]))

        if nome_img_pred in mapa_frames:
            f_real = mapa_frames[nome_img_pred]["frame"]
            t_seg = mapa_frames[nome_img_pred]["timestamp"]
        else:
            nomes_chaves = list(mapa_frames.keys())
            if row_idx < len(nomes_chaves):
                f_real = mapa_frames[nomes_chaves[row_idx]]["frame"]
                t_seg = mapa_frames[nomes_chaves[row_idx]]["timestamp"]
            else:
                f_real = int(row_idx)
                t_seg = float(row_idx * 0.033)

        bx = float(row.get("FaceRectX", 0.0)) if pd.notna(row.get("FaceRectX")) else 0.0
        by = float(row.get("FaceRectY", 0.0)) if pd.notna(row.get("FaceRectY")) else 0.0
        bw = float(row.get("FaceRectWidth", 0.0)) if pd.notna(row.get("FaceRectWidth")) else 0.0
        bh = float(row.get("FaceRectHeight", 0.0)) if pd.notna(row.get("FaceRectHeight")) else 0.0

        emos = row[colunas_emo].astype(float) if colunas_emo else pd.Series()
        score_neutro = float(emos.get("neutral", 0.0))
        emos_sem_neutro = emos.drop(labels=["neutral"], errors="ignore")
        
        emo_ingles = str(emos_sem_neutro.idxmax()) if not emos_sem_neutro.empty else "neutral"
        score_emocao = float(emos_sem_neutro.max()) if not emos_sem_neutro.empty else 0.0

        aus_raw = row[colunas_au].astype(float) if colunas_au else pd.Series()
        aus_delta = (aus_raw - baseline_aus).clip(lower=0.0)

        top_aus = aus_delta.nlargest(3)
        if len(top_aus) >= 3:
            au_pico_val = float(top_aus.iloc[0])
            au_pico_nome = str(top_aus.index[0])
            volume_motor = float((top_aus.iloc[0] * 0.6) + (top_aus.iloc[1] * 0.3) + (top_aus.iloc[2] * 0.1))
        elif len(top_aus) == 2:
            au_pico_val = float(top_aus.iloc[0])
            au_pico_nome = str(top_aus.index[0])
            volume_motor = float((top_aus.iloc[0] * 0.7) + (top_aus.iloc[1] * 0.3))
        elif len(top_aus) == 1:
            au_pico_val = float(top_aus.iloc[0])
            au_pico_nome = str(top_aus.index[0])
            volume_motor = au_pico_val
        else:
            au_pico_val = 0.0
            au_pico_nome = "Nenhuma"
            volume_motor = 0.0

        emo_pt = MAPA_EMOCOES_PT.get(emo_ingles, emo_ingles)

        if score_neutro >= 0.85 and volume_motor < 0.15:
            forca_final = 0.05
            classificacao = "Nivel_0_Basal"
            emo_pt = "neutro"
        else:
            forca_final = float(np.clip((volume_motor * 0.70) + (score_emocao * 0.30), 0.0, 1.0))
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
            "frame": int(f_real),
            "timestamp_segundos": float(t_seg),
            "emocao_predita": str(emo_pt),
            "confianca_emocao": float(round(score_emocao, 3)),
            "grau_neutralidade": float(round(score_neutro, 3)),
            "classificacao": str(classificacao),
            "au_pico": str(au_pico_nome),
            "delta_au_pico": float(round(au_pico_val, 3)),
            "forca_final": float(round(forca_final, 3)),
            "bbox": (bx, by, bw, bh)
        })

    return pd.DataFrame(catalogo)

# ==========================================
# WORKER ASSÍNCRONO EM SEGUNDO PLANO
# ==========================================
async def worker_processar_video_em_background(
    caminho_video_fila: str,
    session_id: str,
    access_key: str,
    selected_emotion: str,
    email: str,
    base_url: str,
    skip_frames: int,
    amostras_por_nivel: int,
    tamanho_imagem: int
):
    async with SEMAFORO_EXECUCAO:
        logger.info(f"[WORKER] Processando job {session_id} para {email}...")
        pasta_frames_temp = None
        
        try:
            pasta_frames_temp, lista_arquivos, mapa_frames, fps = extrair_frames_temporarios(
                caminho_video_fila, skip_frames=skip_frames
            )

            if not lista_arquivos:
                raise ValueError("Erro ao decodificar os quadros do vídeo.")

            loop = asyncio.get_running_loop()
            def rodar_inferencia():
                try:
                    return detector.detect(lista_arquivos)
                except AttributeError:
                    return detector(lista_arquivos)

            res_bruto = await loop.run_in_executor(None, rodar_inferencia)

            df_timeline = analisar_quadros_calibrados(res_bruto, mapa_frames)
            df_validado = df_timeline[df_timeline["emocao_predita"] == selected_emotion].copy()

            # SE A EXPRESSÃO NÃO FOI RECONHECIDA COMO DECLARADA
            if df_validado.empty:
                logger.warning(f"[WORKER] {session_id}: A expressão não bateu com '{selected_emotion}'.")
                colecao_envios.update_one(
                    {"session_id": session_id},
                    {"$set": {
                        "status_processamento": "rejeitado",
                        "motivo_rejeicao": f"Expressão não validada como '{selected_emotion}'."
                    }}
                )

                corpo_rejeicao = f"""
                <html>
                    <body style="font-family: Arial, sans-serif; color: #333; line-height: 1.6;">
                        <div style="max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #eaeaea; border-radius: 8px;">
                            <h3 style="color: #dc2626;">Expressão não identificada</h3>
                            <p>Olá! Obrigado pela sua contribuição com o <strong>GrimCoach</strong>.</p>
                            <p>O algoritmo analisou o vídeo enviado, mas não conseguiu confirmar a expressão de <strong>{selected_emotion.upper()}</strong> com precisão suficiente.</p>
                            <p><strong>Dicas para tentar novamente:</strong></p>
                            <ul>
                                <li>Grave em um local bem iluminado.</li>
                                <li>Evite inclinar a cabeça ou cobrir parte do rosto.</li>
                                <li>Inicie com a face relaxada (neutra) e execute a expressão de forma nítida.</li>
                            </ul>
                            <p>Você pode acessar a plataforma e tentar novamente a qualquer instante!</p>
                        </div>
                    </body>
                </html>
                """
                disparar_email_smtp(email, "GrimCoach - Resultado da Validação do Vídeo", corpo_rejeicao)
                return

            # SE APROVADO:
            primeiro_gatilho = df_validado.iloc[0]
            tempo_primeira_deteccao = {
                "frame": int(primeiro_gatilho["frame"]),
                "segundo": float(primeiro_gatilho["timestamp_segundos"]),
                "nivel_inicial": str(primeiro_gatilho["classificacao"]),
                "forca_inicial": float(primeiro_gatilho["forca_final"])
            }

            # 1. Salva o vídeo na pasta única de vídeos
            ext = os.path.splitext(caminho_video_fila)[1]
            video_hash = calcular_hash_sha256(caminho_video_fila)
            nome_arquivo_video = f"{session_id}{ext}"
            caminho_final_video = os.path.join(DIR_STORAGE_VIDEOS, nome_arquivo_video)
            shutil.move(caminho_video_fila, caminho_final_video)

            url_video_salvo = f"{base_url}static/videos/{nome_arquivo_video}"

            # 2. Recorta e salva faces na pasta única de faces
            cap = cv2.VideoCapture(caminho_final_video)
            imagens_salvas = []

            for nivel, (min_val, max_val) in FAIXAS_NIVEIS.items():
                candidatos = df_validado[(df_validado["forca_final"] >= min_val) & (df_validado["forca_final"] < max_val)].copy()
                if candidatos.empty:
                    continue

                ponto_medio = (min_val + max_val) / 2
                candidatos["dist_centro"] = (candidatos["forca_final"] - ponto_medio).abs()
                selecionados = candidatos.sort_values(by="dist_centro").head(amostras_por_nivel)

                for _, row in selecionados.iterrows():
                    f_num = int(row["frame"])
                    forca = float(row["forca_final"])
                    au = str(row["au_pico"])
                    bbox = row["bbox"]

                    cap.set(cv2.CAP_PROP_POS_FRAMES, f_num)
                    sucesso, frame_bruto = cap.read()
                    if sucesso:
                        rosto = recortar_e_padronizar_rosto(frame_bruto, bbox, tamanho_padrao=tamanho_imagem)
                        
                        id_face = f"face_{uuid.uuid4().hex[:10]}"
                        nome_img = f"{id_face}.png"
                        caminho_img = os.path.join(DIR_STORAGE_FACES, nome_img)
                        cv2.imwrite(caminho_img, rosto)

                        imagens_salvas.append({
                            "id_face": id_face,
                            "nome_arquivo": nome_img,
                            "url": f"{base_url}static/faces/{nome_img}",
                            "caminho_local": str(caminho_img),
                            
                            # Rótulos para a outra API gerenciar
                            "emocao_declarada_inicial": str(selected_emotion),
                            "emocao_predita_ia": str(row["emocao_predita"]),
                            "emocao_consolidada": str(selected_emotion),
                            
                            "nivel": str(nivel),
                            "forca": float(forca),
                            "au_pico": str(au),
                            "frame": int(f_num),
                            "segundo_no_video": float(row["timestamp_segundos"]),
                            
                            "valida": True,
                            "nota_avaliacao": 5,
                            "comentario": "",
                            "atualizado_em": datetime.now(timezone.utc)
                        })

            cap.release()

            # 3. Atualização no MongoDB com os dados estruturados
            colecao_envios.update_one(
                {"session_id": session_id},
                {"$set": {
                    "status_processamento": "concluido",
                    "status_avaliacao": "pendente",
                    "video_file_hash": str(video_hash),
                    "video_url": str(url_video_salvo),
                    "video_local_path": str(caminho_final_video),
                    "video_fps": float(fps),
                    "tempo_primeira_deteccao": tempo_primeira_deteccao,
                    "total_faces_salvas": int(len(imagens_salvas)),
                    "faces": imagens_salvas,
                    "processado_em": datetime.now(timezone.utc)
                }}
            )

            # 4. Envia o e-mail apontando para o seu site frontend
            link_curadoria = f"https://grimcoach.com.br/avaliar?chave={access_key}"
            corpo_sucesso = f"""
            <html>
                <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                    <div style="max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #eaeaea; border-radius: 8px;">
                        <h2 style="color: #2563eb;">Sua expressão foi aprovada!</h2>
                        <p>O seu vídeo de <strong>{selected_emotion.upper()}</strong> gerou {len(imagens_salvas)} recortes faciais catalogados.</p>
                        <p>Acesse o painel para conferir suas fotos e validar cada imagem:</p>
                        <div style="text-align: center; margin: 30px 0;">
                            <a href="{link_curadoria}" style="background-color: #2563eb; color: #ffffff; padding: 12px 24px; text-decoration: none; border-radius: 5px; font-weight: bold;">
                                Avaliar Minhas Fotos
                            </a>
                        </div>
                        <p style="font-size: 13px; color: #666;">Chave única de acesso: <code>{access_key}</code></p>
                    </div>
                </body>
            </html>
            """
            disparar_email_smtp(email, f"GrimCoach - Avalie suas imagens ({selected_emotion.capitalize()})", corpo_sucesso)
            logger.info(f"[WORKER] {session_id} concluído com sucesso.")

        except Exception as e:
            logger.error(f"[WORKER ERRO] Falha crítica no job {session_id}: {str(e)}")
            traceback.print_exc()
            colecao_envios.update_one(
                {"session_id": session_id},
                {"$set": {"status_processamento": "erro", "erro_detalhe": str(e)}}
            )

            corpo_erro = f"""
            <html>
                <body style="font-family: Arial, sans-serif; color: #333; line-height: 1.6;">
                    <div style="max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #eaeaea; border-radius: 8px;">
                        <h3 style="color: #dc2626;">Falha no processamento</h3>
                        <p>Olá! Tivemos uma falha técnica temporária ao analisar o vídeo enviado para <strong>{selected_emotion.upper()}</strong>.</p>
                        <p>Por favor, tente enviar novamente através da plataforma.</p>
                    </div>
                </body>
            </html>
            """
            disparar_email_smtp(email, "GrimCoach - Problema no processamento do vídeo", corpo_erro)

        finally:
            if pasta_frames_temp and os.path.exists(pasta_frames_temp):
                shutil.rmtree(pasta_frames_temp, ignore_errors=True)
            if os.path.exists(caminho_video_fila):
                try:
                    os.remove(caminho_video_fila)
                except OSError:
                    pass

# ==========================================
# ENDPOINT ÚNICO DE RECEBIMENTO (RESPOSTA INSTANTÂNEA)
# ==========================================
@app.post("/api/v1/enviar-video")
async def receber_video_assincrono(
    request: Request,
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    selected_emotion: str = Form(...),
    email: str = Form(...),
    consent_version: str = Form("v1.0-2026"),
    skip_frames: int = Form(2),
    amostras_por_nivel: int = Form(2),
    tamanho_imagem: int = Form(512)
):
    selected_emotion = selected_emotion.strip().lower()
    if selected_emotion not in EMOCOES_PERMITIDAS:
        raise HTTPException(status_code=400, detail="Emoção inválida.")

    email = email.strip().lower()
    if "@" not in email or "." not in email:
        raise HTTPException(status_code=400, detail="E-mail inválido.")

    ext = os.path.splitext(video.filename)[1].lower()
    if ext not in [".mp4", ".mov", ".avi", ".webm"]:
        raise HTTPException(status_code=400, detail="Formato de vídeo não suportado.")

    session_id = f"sub_{uuid.uuid4().hex[:12]}"
    access_key = secrets.token_urlsafe(24)

    # 1. Garante que o diretório da fila existe antes de abrir o arquivo
    caminho_video_fila = os.path.join(DIR_VIDEOS_FILA, f"{session_id}{ext}")
    os.makedirs(os.path.dirname(caminho_video_fila), exist_ok=True)

    with open(caminho_video_fila, "wb") as f_out:
        shutil.copyfileobj(video.file, f_out)

    # 2. Metadados de auditoria LGPD
    client_ip = request.client.host if request.client else "unknown"
    ip_hash = hashlib.sha256(client_ip.encode()).hexdigest()
    user_agent = request.headers.get("user-agent", "unknown")

    # 3. Registra documento inicial no MongoDB
    documento_mongo = {
        "session_id": session_id,
        "access_key": access_key,
        "user_email": email,
        "status_processamento": "na_fila",
        "status_avaliacao": "aguardando_processamento",
        "consent_version": consent_version,
        "consent_timestamp": datetime.now(timezone.utc).isoformat(),
        "ip_address_hash": ip_hash,
        "user_agent": user_agent,
        "selected_emotion": selected_emotion,
        "video_arquivo_original": video.filename,
        "criado_em": datetime.now(timezone.utc),
        "faces": []
    }
    colecao_envios.insert_one(documento_mongo)

    # 4. Dispara a tarefa em segundo plano
    base_url = str(request.base_url)
    background_tasks.add_task(
        worker_processar_video_em_background,
        caminho_video_fila=caminho_video_fila,
        session_id=session_id,
        access_key=access_key,
        selected_emotion=selected_emotion,
        email=email,
        base_url=base_url,
        skip_frames=skip_frames,
        amostras_por_nivel=amostras_por_nivel,
        tamanho_imagem=tamanho_imagem
    )

    logger.info(f"Envio {session_id} aceito para processamento ({email}).")

    # Resposta rápida para a interface do frontend
    return JSONResponse(
        status_code=202,
        content={
            "status": "recebido",
            "email": email
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)