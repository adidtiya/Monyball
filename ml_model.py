import json

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Fitur model valuasi: performa + konteks (usia, menit, liga, posisi)
FITUR_NUM = ['Age', 'Min', 'MP', 'Starts', 'Start_pct', 'Gls_p90', 'npG_p90', 'Ast_p90', 'CrdY_p90', 'PKatt']
FITUR_KAT = ['Asal_Liga', 'Pos_Utama']

# Fitur kemiripan gaya main (tanpa harga/liga agar pemain murah tetap bisa muncul)
FITUR_KEMIRIPAN = ['npG_p90', 'Ast_p90', 'Start_pct']
FITUR_PERAN = ['npG_p90', 'Ast_p90']


def tambah_label_peran(df, n_clusters=4):
    """K-Means pada gol non-penalti & assist per 90 (pemain non-kiper), lalu beri nama tiap kluster."""
    df = df.copy()
    outfield = df['Pos_Utama'] != 'GK'
    X = StandardScaler().fit_transform(df.loc[outfield, FITUR_PERAN])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df.loc[outfield, 'Cluster'] = kmeans.fit_predict(X)
    df.loc[~outfield, 'Cluster'] = -1
    df['Cluster'] = df['Cluster'].astype(int)

    # Nama kluster berdasarkan pusatnya (dalam satuan asli per 90)
    pusat = df[outfield].groupby('Cluster')[FITUR_PERAN].mean()
    total = pusat.sum(axis=1)
    nama = {total.idxmin(): 'Minim kontribusi gol', total.idxmax(): 'Gol + assist tinggi'}
    sisa = pusat.drop(index=list(nama))
    porsi_gol = sisa['npG_p90'] / sisa.sum(axis=1)
    nama[porsi_gol.idxmax()] = 'Pencetak gol'
    nama.update({c: 'Kreator' for c in sisa.index if c not in nama})
    nama[-1] = 'Kiper'
    df['Peran'] = df['Cluster'].map(nama)
    return df


def buat_model_valuasi():
    pre = ColumnTransformer([
        ('num', 'passthrough', FITUR_NUM),
        ('kat', OneHotEncoder(handle_unknown='ignore'), FITUR_KAT),
    ])
    reg = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=42,
    )
    return Pipeline([('pre', pre), ('reg', reg)])


def tambah_prediksi_nilai(df, n_folds=5):
    """Prediksi valuasi wajar tiap pemain dari statistik & konteksnya.

    Prediksi dibuat out-of-fold: tiap pemain dinilai oleh model yang tidak melihat
    harganya sendiri, sehingga selisih prediksi vs harga pasar bisa dibaca sebagai
    indikasi under/overvalued, dan metriknya jujur untuk data baru.
    """
    df = df.copy()
    X = df[FITUR_NUM + FITUR_KAT].fillna(0)
    y = np.log1p(df['Harga'])

    cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    pred_log = cross_val_predict(buat_model_valuasi(), X, y, cv=cv)

    df['Nilai_Prediksi'] = np.expm1(pred_log).round(-4)
    # Catatan: prediksi tertarik ke rata-rata, jadi pemain sangat murah cenderung tampak
    # undervalued dan pemain sangat mahal overvalued. Bandingkan dalam liga/budget yang sama.
    residual = pred_log - y
    df['Undervalued_pct'] = (np.exp(residual) - 1) * 100

    ape = np.abs(df['Nilai_Prediksi'] - df['Harga']) / df['Harga']
    metrik = {
        'n_pemain': int(len(df)),
        'n_folds': n_folds,
        'r2_log': float(r2_score(y, pred_log)),
        'mae_eur': float(mean_absolute_error(df['Harga'], df['Nilai_Prediksi'])),
        'median_ape': float(ape.median()),
        # Selisih di bawah ambang ini masih dalam noise model yang wajar
        'ambang_undervalued_pct': float((np.exp(np.median(np.abs(residual))) - 1) * 100),
        'fitur': FITUR_NUM + FITUR_KAT,
    }
    return df, metrik


def indikasi_nilai(undervalued_pct, ambang):
    if undervalued_pct >= ambang:
        return 'Undervalued'
    if undervalued_pct <= -ambang / (1 + ambang / 100):
        return 'Overvalued'
    return 'Wajar'


def cari_alternatif(df, player_id, filter_pos=True, max_age=None, liga=None, max_harga=None, k=10):
    """Cari k pemain dengan gaya main paling mirip dengan target, setelah filter.

    Kemiripan dihitung pada fitur yang distandarkan terhadap seluruh dataset,
    jadi skornya sebanding antar pencarian. Target tidak pernah masuk hasil.
    """
    target = df.loc[df['player_id'] == player_id].iloc[0]

    pool = df[df['player_id'] != player_id]
    if filter_pos:
        # Pemain cocok jika posisi utama target ada di salah satu posisinya ('MF' cocok dengan 'FWMF')
        pool = pool[pool['Pos'].str.contains(target['Pos_Utama'], na=False)]
    if max_age is not None:
        pool = pool[pool['Age'] <= max_age]
    if liga:
        pool = pool[pool['Asal_Liga'].isin(liga)]
    if max_harga is not None:
        pool = pool[pool['Harga'] <= max_harga]
    if pool.empty:
        return pool.assign(Kemiripan=[])

    X_all = df[FITUR_KEMIRIPAN].fillna(0)
    scaler = StandardScaler().fit(X_all)
    X_pool = scaler.transform(pool[FITUR_KEMIRIPAN].fillna(0))
    x_target = scaler.transform(df.loc[df['player_id'] == player_id, FITUR_KEMIRIPAN].fillna(0))

    knn = NearestNeighbors(n_neighbors=min(k, len(pool))).fit(X_pool)
    dist, idx = knn.kneighbors(x_target)

    hasil = pool.iloc[idx[0]].copy()
    hasil['Kemiripan'] = skor_kemiripan(dist[0], scaler.transform(X_all))
    return hasil


def skor_kemiripan(dist, X_all, n_pasangan=20000):
    """Kemiripan 90% = lebih dekat daripada 90% pasangan pemain acak di dataset."""
    rng = np.random.default_rng(42)
    a, b = rng.integers(0, len(X_all), size=(2, n_pasangan))
    jarak_acak = np.sort(np.linalg.norm(X_all[a] - X_all[b], axis=1))
    return (100 * (1 - np.searchsorted(jarak_acak, dist) / n_pasangan)).round(1)


def load_dataset(path='data/dataset_final_clean.csv', metrics_path='data/model_metrics.json'):
    df = pd.read_csv(path)
    with open(metrics_path, encoding='utf-8') as f:
        metrik = json.load(f)
    return df, metrik


if __name__ == "__main__":
    df, metrik = load_dataset()
    print(f"Model valuasi: R² log-harga {metrik['r2_log']:.2f}, median error {metrik['median_ape'] * 100:.0f}%, "
          f"ambang undervalued {metrik['ambang_undervalued_pct']:.0f}%")

    nama = "Bruno Fernandes"
    target = df[df['Nama'].str.contains(nama, case=False)].iloc[0]
    hasil = cari_alternatif(df, target['player_id'], max_age=30)
    print(f"\nAlternatif untuk {target['Nama']} ({target['Pos']}, {target['Squad']}), usia ≤ 30:")
    for _, p in hasil.iterrows():
        print(f"- {p['Nama']} | {p['Squad']} ({p['Asal_Liga']}) | Mirip {p['Kemiripan']}% | "
              f"Harga €{p['Harga']:,.0f} | Prediksi €{p['Nilai_Prediksi']:,.0f} ({p['Undervalued_pct']:+.0f}%, "
              f"{indikasi_nilai(p['Undervalued_pct'], metrik['ambang_undervalued_pct'])})")
