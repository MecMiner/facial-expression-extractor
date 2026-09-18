import os
import threading
import warnings
import cv2
import tempfile
import shutil
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

try:
    from feat.pipeline import Detector
except ImportError:
    from feat.detector import Detectorv1 as Detector

warnings.filterwarnings("ignore")

DIR_PROJETO = os.path.dirname(os.path.abspath(__file__))
DIR_SAIDA_DATASET = os.path.join(DIR_PROJETO, "dataset_extraido")

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

detector = None

def get_detector():
    global detector
    if detector is None:
        detector = Detector()
    return detector

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
    pasta_temp = tempfile.mkdtemp(prefix="pyfeat_calib_")
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

        # 1. Emoções
        emos = row[colunas_emo].astype(float)
        score_neutro = emos.get("neutral", 0.0)
        emos_sem_neutro = emos.drop(labels=["neutral"], errors="ignore")
        emo_ingles = emos_sem_neutro.idxmax() if not emos_sem_neutro.empty else "neutral"
        score_emocao = emos_sem_neutro.max() if not emos_sem_neutro.empty else 0.0

        # 2. Ativação Muscular Dinâmica (Delta)
        aus_raw = row[colunas_au].astype(float)
        aus_delta = (aus_raw - baseline_aus).clip(lower=0.0)
        
        # Pega as 3 maiores AUs ativas para calcular o volume motor global da face
        top3_aus = aus_delta.nlargest(3)
        au_pico_val = top3_aus.iloc[0] if not top3_aus.empty else 0.0
        au_pico_nome = top3_aus.index[0] if not top3_aus.empty else "Nenhuma"
        volume_motor = (top3_aus.iloc[0] * 0.6) + (top3_aus.iloc[1] * 0.3) + (top3_aus.iloc[2] * 0.1) if len(top3_aus) >= 3 else au_pico_val

        # 3. Força Final Soberana
        if score_neutro >= 0.85 and volume_motor < 0.15:
            forca_final = 0.05
            classificacao = "Neutra Basal"
            emo_pt = "neutro"
        else:
            emo_pt = MAPA_EMOCOES_PT.get(emo_ingles, emo_ingles)
            # O volume motor tem peso predominante (75%) sobre a convicção do classificador (25%)
            forca_final = (volume_motor * 0.75) + (score_emocao * 0.25)
            forca_final = float(np.clip(forca_final, 0.0, 1.0))

            if forca_final < 0.15:
                classificacao = "Nível 0 - Basal"
            elif 0.15 <= forca_final < 0.35:
                classificacao = "Nível 1 - Microexpressão"
            elif 0.35 <= forca_final < 0.58:
                classificacao = "Nível 2 - Sutil"
            elif 0.58 <= forca_final < 0.78:
                classificacao = "Nível 3 - Moderada"
            else:
                classificacao = "Nível 4 - Ápice"

        aus_ativas_dict = aus_delta[aus_delta >= 0.15].round(3).to_dict()

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
            "bbox_h": bh,
            "aus_ativas_delta": str(aus_ativas_dict)
        })

    return pd.DataFrame(catalogo)

def fatiar_video_em_pastas(caminho_video, df_timeline, pasta_destino_base, max_frames_por_nivel=3, tamanho_padrao=512):
    nome_video_sem_ext = os.path.splitext(os.path.basename(caminho_video))[0]
    cap = cv2.VideoCapture(caminho_video)
    frames_extraidos = 0

    emocoes_presentes = df_timeline["emocao_alvo"].unique()

    # Itera primeiro pelos níveis e depois pelas emoções dentro de cada nível
    for nivel, (min_val, max_val) in FAIXAS_NIVEIS.items():
        candidatos_nivel = df_timeline[(df_timeline["forca_final"] >= min_val) & (df_timeline["forca_final"] < max_val)].copy()
        
        if candidatos_nivel.empty:
            continue

        for emo in emocoes_presentes:
            candidatos = candidatos_nivel[candidatos_nivel["emocao_alvo"] == emo].copy()
            if candidatos.empty:
                continue

            # Cria pasta: dataset_extraido / <Nivel> / <Emocao>
            pasta_destino = os.path.join(pasta_destino_base, nivel, emo)
            os.makedirs(pasta_destino, exist_ok=True)

            ponto_medio = (min_val + max_val) / 2
            candidatos["dist_centro"] = (candidatos["forca_final"] - ponto_medio).abs()
            selecionados = candidatos.sort_values(by="dist_centro").head(max_frames_por_nivel)

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
                    nome_img = f"{nome_video_sem_ext}_f{f_num:05d}_{au}_{forca:.2f}.png"
                    cv2.imwrite(os.path.join(pasta_destino, nome_img), rosto)
                    frames_extraidos += 1

    cap.release()
    return frames_extraidos

class AppAnalisador(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Analisador FACS - Organização por Emoção e Nível")
        self.geometry("720x600")
        self.resizable(False, False)

        self.skip_frames_var = tk.IntVar(value=3)
        self.extrair_frames_var = tk.BooleanVar(value=True)
        self.frames_por_nivel_var = tk.IntVar(value=3)
        self.tamanho_padrao_var = tk.IntVar(value=512)

        self._criar_interface()

    def _criar_interface(self):
        frame_arquivo = ttk.LabelFrame(self, text=" 1. Seleção do Vídeo ", padding=10)
        frame_arquivo.pack(fill="x", padx=15, pady=8)

        self.entry_arquivo = ttk.Entry(frame_arquivo, width=55)
        self.entry_arquivo.pack(side="left", padx=5, fill="x", expand=True)

        btn_arquivo = ttk.Button(frame_arquivo, text="Buscar Vídeo...", command=self._selecionar_arquivo)
        btn_arquivo.pack(side="right", padx=5)

        frame_cfg = ttk.LabelFrame(self, text=" 2. Configurações de Análise e Dataset ", padding=10)
        frame_cfg.pack(fill="x", padx=15, pady=5)

        f_skip = ttk.Frame(frame_cfg)
        f_skip.pack(fill="x", pady=3)
        ttk.Label(f_skip, text="Salto de frames (skip_frames):").pack(side="left")
        self.sp_frames = ttk.Spinbox(f_skip, from_=1, to=30, textvariable=self.skip_frames_var, width=5)
        self.sp_frames.pack(side="left", padx=8)
        ttk.Label(f_skip, text="(3 = precisão ideal para capturar o ápice)").pack(side="left")

        f_fatiar = ttk.Frame(frame_cfg)
        f_fatiar.pack(fill="x", pady=3)
        self.chk_extrair = ttk.Checkbutton(f_fatiar, text="Extrair para [Emoção / Nível]", variable=self.extrair_frames_var)
        self.chk_extrair.pack(side="left")

        ttk.Label(f_fatiar, text="Amostras p/ nível:").pack(side="left", padx=(15, 5))
        self.sp_amostras = ttk.Spinbox(f_fatiar, from_=1, to=10, textvariable=self.frames_por_nivel_var, width=4)
        self.sp_amostras.pack(side="left")

        f_res = ttk.Frame(frame_cfg)
        f_res.pack(fill="x", pady=4)
        ttk.Label(f_res, text="Resolução da Face:").pack(side="left")
        cb_res = ttk.Combobox(f_res, textvariable=self.tamanho_padrao_var, values=[256, 384, 512, 768], width=6, state="readonly")
        cb_res.pack(side="left", padx=8)
        ttk.Label(f_res, text="px (1:1 perfeitamente centralizado)").pack(side="left")

        # Na função _criar_interface:
        lbl_info = ttk.Label(frame_cfg, text="Destino: ./dataset_extraido/<nivel>/<emocao>/", font=("Segoe UI", 8, "italic"))
        lbl_info.pack(anchor="w", pady=(4, 0))

        frame_acao = ttk.Frame(self, padding=8)
        frame_acao.pack(fill="x", padx=15)

        self.btn_iniciar = ttk.Button(frame_acao, text="Analisar, Categorizar e Gerar Dataset", command=self._iniciar_thread)
        self.btn_iniciar.pack(fill="x", pady=5)

        self.progresso = ttk.Progressbar(frame_acao, mode="indeterminate")
        self.progresso.pack(fill="x", pady=5)

        frame_log = ttk.LabelFrame(self, text=" Log de Execução ", padding=10)
        frame_log.pack(fill="both", expand=True, padx=15, pady=8)

        self.txt_log = tk.Text(frame_log, height=10, state="disabled", wrap="word")
        self.txt_log.pack(fill="both", expand=True)

    def _selecionar_arquivo(self):
        tipos = [("Arquivos de Vídeo", "*.mp4 *.avi *.mov *.mkv"), ("Todos os Arquivos", "*.*")]
        caminho = filedialog.askopenfilename(filetypes=tipos)
        if caminho:
            self.entry_arquivo.delete(0, tk.END)
            self.entry_arquivo.insert(0, caminho)

    def _log(self, texto):
        self.txt_log.configure(state="normal")
        self.txt_log.insert(tk.END, texto + "\n")
        self.txt_log.see(tk.END)
        self.txt_log.configure(state="disabled")

    def _iniciar_thread(self):
        arquivo = self.entry_arquivo.get().strip()
        if not arquivo or not os.path.isfile(arquivo):
            messagebox.showwarning("Aviso", "Selecione um arquivo de vídeo válido.")
            return

        self.btn_iniciar.configure(state="disabled")
        self.progresso.start(10)
        threading.Thread(target=self._executar_processamento, daemon=True).start()

    def _executar_processamento(self):
        caminho_video = self.entry_arquivo.get().strip()
        nome_vid = os.path.basename(caminho_video)
        nome_vid_sem_ext = os.path.splitext(nome_vid)[0]
        pasta_temp = None

        try:
            os.makedirs(DIR_SAIDA_DATASET, exist_ok=True)
            skip = self.skip_frames_var.get()
            extrair = self.extrair_frames_var.get()
            max_amostras = self.frames_por_nivel_var.get()
            tam_padrao = self.tamanho_padrao_var.get()

            self._log(f"\nIniciando: {nome_vid}")
            pasta_temp, lista_arquivos, indices_frames, total_frames = extrair_frames_temporarios(caminho_video, skip_frames=skip)

            if not lista_arquivos:
                self._log("[!] Erro ao decodificar quadros.")
                return

            self._log(f"-> {len(lista_arquivos)} quadros enviados para o detector.")
            dt = get_detector()

            try:
                res_bruto = dt.detect(lista_arquivos)
            except AttributeError:
                res_bruto = dt(lista_arquivos)

            self._log("Calculando contração motora e volume FACS...")
            df_calibrado = processar_com_calibracao_baseline(res_bruto, indices_frames, prefixo_origem=nome_vid)
            df_calibrado["video_origem"] = nome_vid

            csv_timeline = os.path.join(DIR_SAIDA_DATASET, f"{nome_vid_sem_ext}_timeline.csv")
            df_calibrado.to_csv(csv_timeline, index=False, encoding="utf-8-sig")
            self._log(f"[OK] Timeline salva: dataset_extraido/{os.path.basename(csv_timeline)}")

            if extrair:
                self._log("Organizando imagens por [Emoção / Nível]...")
                qtd = fatiar_video_em_pastas(
                    caminho_video, df_calibrado, DIR_SAIDA_DATASET,
                    max_frames_por_nivel=max_amostras, tamanho_padrao=tam_padrao
                )
                self._log(f"[OK] {qtd} imagens centralizadas salvas com sucesso.")

            self._log("\n[SUCESSO] Processamento e categorização finalizados!")
            messagebox.showinfo("Sucesso", f"Imagens organizadas em:\n{DIR_SAIDA_DATASET}")

        except Exception as e:
            self._log(f"\n[ERRO] Falha: {e}")
            messagebox.showerror("Erro", f"Ocorreu um erro: {e}")

        finally:
            if pasta_temp and os.path.exists(pasta_temp):
                shutil.rmtree(pasta_temp, ignore_errors=True)
            self.progresso.stop()
            self.btn_iniciar.configure(state="normal")

if __name__ == "__main__":
    app = AppAnalisador()
    app.mainloop()