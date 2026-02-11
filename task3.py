"""
Task 3 — Benchmark (tiempos) + Tabla para decidir entre SIFT vs ORB

Input esperado en la misma carpeta:
- fotofrontal.jpeg
- fotorotada.jpeg

Salida:
- benchmark_results.csv
- (opcional) prints de la tabla en consola

Requisito para SIFT:
pip install opencv-contrib-python
"""

import cv2
import numpy as np
import time
import csv

FRONTAL = "fotofrontal.jpeg"
ROTADA  = "fotorotada.jpeg"
RATIO_THRESH = 0.75

# Repeticiones para promediar tiempos (sube si quieres más estabilidad)
WARMUP_RUNS = 3
MEASURE_RUNS = 20


def load_gray(path: str):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer: {path}. Verifica nombre/ubicación.")
    return img


def knn_ratio_test(knn_matches, ratio=0.75):
    good = []
    for pair in knn_matches:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


def ms_from_perf(start, end):
    return (end - start) * 1000.0


def timed_detect_and_compute(detector, img):
    """
    Mide SOLO detección+descripción: detectAndCompute.
    Retorna: (kp, des, time_ms)
    """
    t0 = time.perf_counter()
    kp, des = detector.detectAndCompute(img, None)
    t1 = time.perf_counter()
    return kp, des, ms_from_perf(t0, t1)


def timed_knn_match(matcher, des1, des2, k=2):
    """
    Mide SOLO matching (knnMatch).
    Retorna: (knn_matches, time_ms)
    """
    t0 = time.perf_counter()
    knn = matcher.knnMatch(des1, des2, k=k)
    t1 = time.perf_counter()
    return knn, ms_from_perf(t0, t1)


def benchmark_sift(img1, img2):
    """
    SIFT + BFMatcher L2 + Ratio Test
    Reporta tiempos promedio y conteos.
    """
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    # Warmup (para estabilizar caches/jit interno)
    for _ in range(WARMUP_RUNS):
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        if des1 is not None and des2 is not None:
            _ = bf.knnMatch(des1, des2, k=2)

    det_times = []
    match_times = []
    last_counts = None

    for _ in range(MEASURE_RUNS):
        kp1, des1, t_det1 = timed_detect_and_compute(sift, img1)
        kp2, des2, t_det2 = timed_detect_and_compute(sift, img2)

        # Si no hay descriptores, no se puede medir matching
        if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
            knn = []
            t_match = 0.0
            good = []
        else:
            knn, t_match = timed_knn_match(bf, des1, des2, k=2)
            good = knn_ratio_test(knn, ratio=RATIO_THRESH)

        det_times.append(t_det1 + t_det2)
        match_times.append(t_match)

        last_counts = (len(kp1), len(kp2), len(good))

    det_avg = float(np.mean(det_times))
    match_avg = float(np.mean(match_times))
    total_avg = det_avg + match_avg

    kp1c, kp2c, goodc = last_counts
    return {
        "Algoritmo": "SIFT",
        "Det+Desc_ms_avg": det_avg,
        "Matching_ms_avg": match_avg,
        "Tiempo_total_ms_avg": total_avg,
        "Keypoints_A": kp1c,
        "Keypoints_B": kp2c,
        "Good_matches": goodc,
    }


def benchmark_orb(img1, img2):
    """
    ORB + BFMatcher Hamming + Ratio Test
    Reporta tiempos promedio y conteos.
    """
    orb = cv2.ORB_create(nfeatures=2000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # Warmup
    for _ in range(WARMUP_RUNS):
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        if des1 is not None and des2 is not None:
            _ = bf.knnMatch(des1, des2, k=2)

    det_times = []
    match_times = []
    last_counts = None

    for _ in range(MEASURE_RUNS):
        kp1, des1, t_det1 = timed_detect_and_compute(orb, img1)
        kp2, des2, t_det2 = timed_detect_and_compute(orb, img2)

        if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
            knn = []
            t_match = 0.0
            good = []
        else:
            knn, t_match = timed_knn_match(bf, des1, des2, k=2)
            good = knn_ratio_test(knn, ratio=RATIO_THRESH)

        det_times.append(t_det1 + t_det2)
        match_times.append(t_match)

        last_counts = (len(kp1), len(kp2), len(good))

    det_avg = float(np.mean(det_times))
    match_avg = float(np.mean(match_times))
    total_avg = det_avg + match_avg

    kp1c, kp2c, goodc = last_counts
    return {
        "Algoritmo": "ORB",
        "Det+Desc_ms_avg": det_avg,
        "Matching_ms_avg": match_avg,
        "Tiempo_total_ms_avg": total_avg,
        "Keypoints_A": kp1c,
        "Keypoints_B": kp2c,
        "Good_matches": goodc,
    }


def print_table(rows):
    # impresión simple sin pandas
    headers = [
        "Algoritmo",
        "Tiempo_total_ms_avg",
        "Keypoints_A",
        "Keypoints_B",
        "Good_matches",
        "Det+Desc_ms_avg",
        "Matching_ms_avg",
    ]
    colw = {h: max(len(h), max(len(f"{r[h]:.2f}") if isinstance(r[h], float) else len(str(r[h])) for r in rows)) for h in headers}

    def fmt_cell(h, v):
        if isinstance(v, float):
            s = f"{v:.2f}"
        else:
            s = str(v)
        return s.ljust(colw[h])

    line = " | ".join(h.ljust(colw[h]) for h in headers)
    sep  = "-+-".join("-" * colw[h] for h in headers)
    print(line)
    print(sep)
    for r in rows:
        print(" | ".join(fmt_cell(h, r[h]) for h in headers))


def save_csv(rows, path="benchmark_results.csv"):
    headers = [
        "Algoritmo",
        "Tiempo_total_ms_avg",
        "Keypoints_A",
        "Keypoints_B",
        "Good_matches",
        "Det+Desc_ms_avg",
        "Matching_ms_avg",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    img1 = load_gray(FRONTAL)
    img2 = load_gray(ROTADA)

    print("== Benchmark Task 3 ==")
    print(f"Imágenes: {FRONTAL} {img1.shape} | {ROTADA} {img2.shape}")
    print(f"WARMUP_RUNS={WARMUP_RUNS}, MEASURE_RUNS={MEASURE_RUNS}, RATIO={RATIO_THRESH}")

    sift_res = benchmark_sift(img1, img2)
    orb_res  = benchmark_orb(img1, img2)

    rows = [sift_res, orb_res]

    print("\n== Resultados (promedios en ms) ==")
    print_table(rows)

    save_csv(rows, "benchmark_results.csv")
    print("\nSe guardó: benchmark_results.csv")

    # Extra útil para tu análisis del producto A (presupuesto ~16ms)
    for r in rows:
        fps_est = 1000.0 / r["Tiempo_total_ms_avg"] if r["Tiempo_total_ms_avg"] > 0 else 0.0
        print(f"{r['Algoritmo']}: tiempo_total={r['Tiempo_total_ms_avg']:.2f} ms -> FPS aprox={fps_est:.1f}")

    print("\nListo ✅")


if __name__ == "__main__":
    main()
