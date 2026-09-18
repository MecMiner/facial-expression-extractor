from feat.detector import Detectorv1

print("Inicializando detector Py-Feat...")

try:
    detector = Detectorv1(
        face_model="retinaface",
        landmark_model="mobilenet",
        au_model="xgb",
        emotion_model="resmasknet"
    )
    print("\n[OK] Detector carregado com sucesso!")
except Exception as e:
    print(f"\n[ERRO] Falha ao inicializar o detector: {e}")