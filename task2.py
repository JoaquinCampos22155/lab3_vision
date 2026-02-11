"""
Correspondencia completa con SIFT y ORB (OpenCV)
Input esperado en la misma carpeta:
- fotofrontal.jpeg
- fotorotada.jpeg

Salida:
- matches_sift_inliers.jpg
- matches_orb_inliers.jpg

Requisitos:
pip install opencv-contrib-python
"""

import cv2
import numpy as np
import os


FRONTAL = "fotofrontal.jpeg"
ROTADA  = "fotorotada.jpeg"
RATIO_THRESH = 0.75


def load_gray(path: str):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer: {path}. Verifica nombre/ubicación.")
    return img


def knn_ratio_test(knn_matches, ratio=0.75):
    """
    Lowe's ratio test:
    conservar match m si m.distance < ratio * n.distance (m=mejor vecino, n=2do vecino)
    """
    good = []
    for pair in knn_matches:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


def ransac_inliers(kp1, kp2, good_matches, reproj_thresh=3.0):
    """
    Estima homografía con RANSAC y devuelve:
    - mask_inliers (lista 0/1)
    - H (homografía)
    - inlier_matches (lista de matches que sí son inliers)
    """
    if len(good_matches) < 4:
        return None, None, []

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, reproj_thresh)
    if mask is None:
        return None, H, []

    mask = mask.ravel().tolist()
    inliers = [m for m, keep in zip(good_matches, mask) if keep]
    return mask, H, inliers


def draw_inliers(img1, kp1, img2, kp2, inlier_matches, title="inliers"):
    """
    Dibuja SOLO las líneas de correspondencia de los matches inliers.
    """
    out = cv2.drawMatches(
        img1, kp1,
        img2, kp2,
        inlier_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # Opcional: escribir texto con conteos
    cv2.putText(out, f"{title}: {len(inlier_matches)} inliers",
                (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def run_sift(img1, img2):
    # 2) SIFT detect + describe
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        raise RuntimeError("SIFT no encontró descriptores suficientes en alguna imagen.")

    # 4a) Matching SIFT con BFMatcher L2 (Euclidiana)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    # KNN para ratio test
    knn = bf.knnMatch(des1, des2, k=2)

    # 5) Lowe ratio test
    good = knn_ratio_test(knn, ratio=RATIO_THRESH)

    # 6) Inliers con RANSAC (homografía) y dibujar SOLO inliers
    _, _, inliers = ransac_inliers(kp1, kp2, good, reproj_thresh=3.0)

    out = draw_inliers(img1, kp1, img2, kp2, inliers, title="SIFT")
    return out, len(kp1), len(kp2), len(good), len(inliers)


def run_orb(img1, img2):
    # 3) ORB detect + describe
    orb = cv2.ORB_create(nfeatures=2000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        raise RuntimeError("ORB no encontró descriptores suficientes en alguna imagen.")

    # 4b) Matching ORB con BFMatcher Hamming
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # KNN para ratio test
    knn = bf.knnMatch(des1, des2, k=2)

    # 5) Lowe ratio test (también aplica para ORB)
    good = knn_ratio_test(knn, ratio=RATIO_THRESH)

    # 6) Inliers con RANSAC (homografía) y dibujar SOLO inliers
    _, _, inliers = ransac_inliers(kp1, kp2, good, reproj_thresh=3.0)

    out = draw_inliers(img1, kp1, img2, kp2, inliers, title="ORB")
    return out, len(kp1), len(kp2), len(good), len(inliers)


def main():
    # 1) Cargar imágenes en grises
    img1 = load_gray(FRONTAL)
    img2 = load_gray(ROTADA)

    print("== Cargando imágenes ==")
    print(f"- {FRONTAL}: {img1.shape}")
    print(f"- {ROTADA}:  {img2.shape}")

    print("\n== SIFT ==")
    sift_img, kp1c, kp2c, goodc, inlc = run_sift(img1, img2)
    print(f"Keypoints img1: {kp1c}")
    print(f"Keypoints img2: {kp2c}")
    print(f"Good matches (ratio test): {goodc}")
    print(f"Inliers (RANSAC): {inlc}")

    print("\n== ORB ==")
    orb_img, kp1c, kp2c, goodc, inlc = run_orb(img1, img2)
    print(f"Keypoints img1: {kp1c}")
    print(f"Keypoints img2: {kp2c}")
    print(f"Good matches (ratio test): {goodc}")
    print(f"Inliers (RANSAC): {inlc}")

    # Guardar resultados
    cv2.imwrite("matches_sift_inliers.jpg", sift_img)
    cv2.imwrite("matches_orb_inliers.jpg", orb_img)

    print("\nListo ✅")
    print("Se generaron:")
    print("- matches_sift_inliers.jpg")
    print("- matches_orb_inliers.jpg")

    # Mostrar (opcional). Cierra con cualquier tecla.
    cv2.imshow("SIFT Inliers", sift_img)
    cv2.imshow("ORB Inliers", orb_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
