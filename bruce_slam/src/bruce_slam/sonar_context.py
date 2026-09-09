"""SONAR Context + Polar Key — descripteurs de place recognition (Kim et al., ICRA 2023).

Module pur NumPy (aucune dépendance ROS) → testable seul sur des images PNG.

Convention d'axes FIXÉE pour tout le module :
    - axe 0 (lignes)   = AZIMUTH  (A directions angulaires)
    - axe 1 (colonnes) = RANGE    (R distances)
Une inversion casse silencieusement le matching → ne pas changer sans tout vérifier.

Pipeline :
    image sonar  --build_sonar_context-->  contexte (A x R)  --build_polar_key-->  polar key (R,)
    matching : cosine_distance_shifted(contexte_q, contexte_c) -> (distance, shift_azimuth, shift_range)
"""
import numpy as np


def build_sonar_context(image, num_azimuth=40, num_range=40, intensity_threshold=95):
    """Construit le SONAR Context : matrice (num_azimuth x num_range).

    Encode la DENSITÉ DES RETOURS STRUCTURELS : par case, moyenne de l'intensité
    AU-DESSUS d'un seuil (max(intensité - seuil, 0)). Une case forte = beaucoup
    de réflecteurs puissants → signature géométrique du lieu.

    Pourquoi pas le max-pooling de l'intensité brute (formulation d'origine du
    papier) : sur le P900 (basse résolution, fort fond de mer, retours proches
    saturants), le max sature partout → descripteur quasi uniforme, non
    discriminant. Mesuré sur sc_descriptor_bench.py : max brut AUC=0.55 (aléatoire)
    vs densité au-dessus du seuil AUC=0.86 (vraies/fausses revisites bien séparées).

    En mode cartésien Aracati, l'image fournie est déjà REPOLARISÉE
    (lignes = range, colonnes = azimuth ; cf. feature_extraction._polar_remap).

    Args:
        image              : image sonar 2D (H x W), niveaux de gris.
        num_azimuth        : A, nombre de cases en azimuth (sortie).
        num_range          : R, nombre de cases en range (sortie).
        intensity_threshold: seuil sous lequel l'intensité est ignorée (fond/bruit).

    Returns:
        contexte (num_azimuth, num_range) en float [0, 1].
    """
    img = np.asarray(image, dtype=np.float32)
    if img.ndim != 2:
        raise ValueError("build_sonar_context attend une image 2D en niveaux de gris")

    # On ne garde que l'intensité au-dessus du seuil (retours structurels).
    if intensity_threshold > 0:
        img = np.clip(img - float(intensity_threshold), 0.0, None)

    # On veut que l'axe AZIMUTH soit l'axe 0 du contexte. Dans l'image
    # repolarisée, les colonnes (W) = azimuth et les lignes (H) = range.
    # On transpose donc pour avoir (azimuth, range) = (W, H) avant le pooling.
    img = img.T  # maintenant (W=azimuth, H=range)
    A_src, R_src = img.shape

    # Mean-pooling par blocs vers (num_azimuth, num_range) : la MOYENNE (densité)
    # discrimine, là où le max saturait. On rogne au multiple inférieur.
    a_blk = max(A_src // num_azimuth, 1)
    r_blk = max(R_src // num_range, 1)
    cropped = img[:a_blk * num_azimuth, :r_blk * num_range]
    pooled = cropped.reshape(num_azimuth, a_blk, num_range, r_blk).mean(axis=(1, 3))

    # Normalisation en [0, 1] pour stabiliser la distance cosinus
    m = pooled.max()
    if m > 0:
        pooled = pooled / m
    return pooled.astype(np.float32)


def build_polar_key(context):
    """Construit la Polar Key : vecteur 1D de dimension R (range).

    Résume chaque distance (range) par la moyenne d'intensité sur tout l'azimuth
    (formule P_j = moyenne de la ligne range j). Sert à la recherche rapide
    par KD-tree (distance euclidienne).

    Args:
        context : matrice SONAR Context (A x R).

    Returns:
        polar_key (R,) : moyenne sur l'axe azimuth (axe 0).
    """
    return np.asarray(context, dtype=np.float32).mean(axis=0)


def cosine_distance_columns(ctx_q, ctx_c):
    """Distance cosinus colonne par colonne entre deux contextes (même forme A x R).

    Formule : D = (1/A) * somme_j (1 - cos(colonne_j^q, colonne_j^c)).
    Une colonne nulle (norme 0) contribue 1.0 (distance maximale) → évite la
    division par zéro et pénalise l'absence d'information.

    Returns:
        distance scalaire dans [0, 1] (0 = identique, 1 = orthogonal/vide).
    """
    q = np.asarray(ctx_q, dtype=np.float32)
    c = np.asarray(ctx_c, dtype=np.float32)
    # normes par colonne (axe 0 = azimuth → on parcourt les colonnes = range)
    nq = np.linalg.norm(q, axis=0)
    nc = np.linalg.norm(c, axis=0)
    dots = np.sum(q * c, axis=0)
    R = q.shape[1]
    dist_cols = np.ones(R, dtype=np.float32)  # par défaut 1.0 (colonne vide)
    valid = (nq > 1e-9) & (nc > 1e-9)
    dist_cols[valid] = 1.0 - dots[valid] / (nq[valid] * nc[valid])
    return float(dist_cols.mean())


def _shift_zero_pad(mat, shift, axis):
    """Décale `mat` de `shift` le long de `axis` avec ZERO PADDING (pas circulaire).

    Le FOV du sonar est limité : ce qui sort du champ n'existe pas → on remplit
    de zéros au lieu de faire un wrap-around (qui inventerait de fausses voisines).
    """
    out = np.zeros_like(mat)
    if shift == 0:
        return mat.copy()
    if axis == 0:
        if shift > 0:
            out[shift:, :] = mat[:-shift, :]
        else:
            out[:shift, :] = mat[-shift:, :]
    else:
        if shift > 0:
            out[:, shift:] = mat[:, :-shift]
        else:
            out[:, :shift] = mat[:, -shift:]
    return out


def cosine_distance_shifted(ctx_q, ctx_c, max_col_shift=10, max_row_shift=5):
    """Distance cosinus minimale sur tous les décalages bornés (adaptive shifting).

    On teste tous les décalages :
      - colonnes (= azimuth, axe 0) dans [-max_col_shift, +max_col_shift] → rotation
      - lignes   (= range,   axe 1) dans [-max_row_shift, +max_row_shift] → translation
    avec zero padding. On garde le décalage qui MINIMISE la distance : c'est le
    meilleur alignement entre les deux passages du même lieu.

    NOTE : les noms suivent l'image (ctx en A x R). "col_shift" décale l'azimuth
    (axe 0), "row_shift" décale le range (axe 1).

    Returns:
        (best_distance, best_azimuth_shift, best_range_shift)
        Le shift optimal donne aussi une estimation grossière de la pose relative
        (utilisable plus tard comme init ICP).
    """
    q = np.asarray(ctx_q, dtype=np.float32)
    c = np.asarray(ctx_c, dtype=np.float32)
    best = (np.inf, 0, 0)
    for sa in range(-max_col_shift, max_col_shift + 1):
        c_a = _shift_zero_pad(c, sa, axis=0)  # décalage azimuth
        for sr in range(-max_row_shift, max_row_shift + 1):
            c_ar = _shift_zero_pad(c_a, sr, axis=1)  # décalage range
            d = cosine_distance_columns(q, c_ar)
            if d < best[0]:
                best = (d, sa, sr)
    return best


if __name__ == "__main__":
    # Petit test manuel : deux images identiques → distance ~0 ; décalée → retrouvée.
    import sys
    rng = np.random.default_rng(0)
    img = (rng.random((512, 512)) * 255).astype(np.uint8)
    ctx = build_sonar_context(img, 40, 40)
    pkey = build_polar_key(ctx)
    print("contexte:", ctx.shape, " polar key:", pkey.shape)
    d_same, *_ = cosine_distance_shifted(ctx, ctx, 5, 5)
    print("distance image vs elle-même:", round(d_same, 4), "(attendu ~0)")
    # image décalée en azimuth de 3 → l'adaptive shifting doit la retrouver
    shifted = _shift_zero_pad(ctx, 3, axis=0)
    d, sa, sr = cosine_distance_shifted(ctx, shifted, 5, 5)
    print(f"distance image décalée: {d:.4f}, shift trouvé azimuth={sa} range={sr} (attendu azimuth=-3)")
