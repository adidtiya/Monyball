import json
import re
import unicodedata
from collections import defaultdict

import pandas as pd

from ml_model import tambah_label_peran, tambah_prediksi_nilai

# File FBref -> (nama liga, kode kompetisi Transfermarkt)
FBREF_FILES = {
    'data/data_fbref_standard.csv': ('Premier League', 'GB1'),
    'data/data_fbref_serieA.csv': ('Serie A', 'IT1'),
    'data/data_fbref_LaLiga.csv': ('La Liga', 'ES1'),
    'data/data_fbref_eredivisie.csv': ('Eredivisie', 'NL1'),
    'data/data_fbref_belgian.csv': ('Belgian Pro League', 'BE1'),
    'data/data_fbref_portugal.csv': ('Liga Portugal', 'PO1'),
    'data/data_fbref_J1.csv': ('J1 League', 'JAP1'),
    'data/data_fbref_aus.csv': ('A-League', 'AUS1'),
    'data/data_fbref_mls.csv': ('MLS', 'MLS1'),
}

KOLOM_ANGKA = ['Age', 'Born', 'MP', 'Starts', 'Min', 'Gls', 'Ast', 'G-PK', 'PK', 'PKatt', 'CrdY', 'CrdR']
KOLOM_JUMLAH = ['MP', 'Starts', 'Min', 'Gls', 'Ast', 'G-PK', 'PK', 'PKatt', 'CrdY', 'CrdR']
MIN_MENIT = 900
# Valuasi yang lebih tua dari ini dianggap basi untuk statistik musim berjalan
MAKS_UMUR_VALUASI_HARI = 540


def normalisasi_nama(nama):
    """'Marc Guéhi' -> {'marc', 'guehi'}; urutan kata & aksen diabaikan."""
    s = unicodedata.normalize('NFKD', str(nama)).encode('ascii', 'ignore').decode().lower()
    s = re.sub(r"['’`.]", '', s)
    s = re.sub(r'[^a-z ]', ' ', s)
    return frozenset(s.split())


def load_fbref():
    dfs = []
    for file_path, (liga, kode_tm) in FBREF_FILES.items():
        # Kolom ke-2 baris header berisi nama kolom; baris 'Rk' berulang (jika ada) dibuang
        df = pd.read_csv(file_path, header=1)
        df.columns = df.columns.str.strip()
        df = df[df['Rk'].astype(str) != 'Rk']
        df = df.rename(columns={'Player': 'Nama'})[['Nama', 'Nation', 'Pos', 'Squad'] + KOLOM_ANGKA]
        df['Asal_Liga'] = liga
        df['Kode_Liga_TM'] = kode_tm
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    for col in KOLOM_ANGKA:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
    df[KOLOM_JUMLAH] = df[KOLOM_JUMLAH].fillna(0)
    return df


def load_transfermarkt():
    df = pd.read_csv('data/players.csv', parse_dates=['date_of_birth'])
    df = df[df['market_value_in_eur'].notna()].copy()

    # Tanggal valuasi terakhir per pemain (market_value_in_eur = valuasi terakhir)
    val = pd.read_csv('data/player_valuations.csv', usecols=['player_id', 'date'], parse_dates=['date'])
    df = df.merge(val.groupby('player_id')['date'].max().rename('Tgl_Valuasi'), on='player_id', how='left')

    df['Tahun_Lahir'] = df['date_of_birth'].dt.year
    df['Token'] = df['name'].map(normalisasi_nama)
    return df


def cocokkan_pemain(df_fbref, df_tm):
    """Pasangkan baris FBref ke player_id Transfermarkt memakai nama + tahun lahir.

    1. Nama sama persis (setelah normalisasi).
    2. Nama satu adalah bagian dari nama lain ('Dennis Eckert' vs 'Dennis Eckert Ayensa'),
       hanya jika ≥2 kata cocok atau pemain TM memang bermain di liga yang sama.
    Kandidat ganda diselesaikan dengan liga; jika tetap ganda, baris dilewati.
    """
    per_tahun = defaultdict(list)
    for p in df_tm[['player_id', 'Token', 'Tahun_Lahir', 'current_club_domestic_competition_id']].itertuples(index=False):
        per_tahun[p.Tahun_Lahir].append(p)

    def pilih(kandidat, kode_liga):
        if len(kandidat) > 1:
            kandidat = [p for p in kandidat if p.current_club_domestic_competition_id == kode_liga]
        return kandidat[0].player_id if len(kandidat) == 1 else None

    hasil, metode = [], []
    for r in df_fbref[['Nama', 'Born', 'Kode_Liga_TM']].itertuples(index=False):
        token = normalisasi_nama(r.Nama)
        kandidat_tahun = per_tahun.get(r.Born, []) if pd.notna(r.Born) else []

        pid = pilih([p for p in kandidat_tahun if p.Token == token], r.Kode_Liga_TM)
        cara = 'persis'
        if pid is None:
            sebagian = [
                p for p in kandidat_tahun
                if (token <= p.Token or p.Token <= token)
                and (len(token & p.Token) >= 2 or p.current_club_domestic_competition_id == r.Kode_Liga_TM)
            ]
            pid = pilih(sebagian, r.Kode_Liga_TM)
            cara = 'sebagian'
        hasil.append(pid)
        metode.append(cara if pid is not None else 'gagal')

    df = df_fbref.copy()
    df['player_id'] = pd.array(hasil, dtype='Int64')
    df['Metode_Cocok'] = metode
    return df


def gabung_pemain_pindah(df):
    """Pemain yang pindah klub di tengah musim muncul >1 baris; jumlahkan statistiknya."""
    df = df.sort_values('Min', ascending=False)
    agg = {col: 'sum' for col in KOLOM_JUMLAH}
    # Identitas, posisi, dan liga diambil dari klub dengan menit terbanyak
    agg.update({col: 'first' for col in ['Nama', 'Nation', 'Pos', 'Asal_Liga', 'Age', 'Born', 'Metode_Cocok']})
    gabung = df.groupby('player_id').agg(agg)
    gabung['Squad'] = df.groupby('player_id')['Squad'].agg(' / '.join)
    return gabung.reset_index()


def hitung_fitur(df):
    n90 = df['Min'] / 90
    df['Gls_p90'] = df['Gls'] / n90
    df['Ast_p90'] = df['Ast'] / n90
    df['npG_p90'] = df['G-PK'] / n90
    df['GA_p90'] = (df['Gls'] + df['Ast']) / n90
    df['CrdY_p90'] = df['CrdY'] / n90
    df['Start_pct'] = df['Starts'] / df['MP'].where(df['MP'] > 0)
    # FBref menulis posisi ganda tanpa pemisah: 'MFFW' = MF utama, FW kedua
    df['Pos_Utama'] = df['Pos'].str[:2]
    return df


def process_data():
    print("Memulai proses pembersihan dan penggabungan dataset...")

    df_fbref = load_fbref()
    df_tm = load_transfermarkt()
    print(f"FBref: {len(df_fbref)} baris dari {df_fbref['Asal_Liga'].nunique()} liga.")

    df = cocokkan_pemain(df_fbref, df_tm)
    print("Hasil pencocokan ke Transfermarkt:", df['Metode_Cocok'].value_counts().to_dict())
    df = df[df['player_id'].notna()]

    df = gabung_pemain_pindah(df)
    df = df.merge(
        df_tm[['player_id', 'name', 'date_of_birth', 'market_value_in_eur', 'Tgl_Valuasi']],
        on='player_id', how='inner',
    ).rename(columns={'market_value_in_eur': 'Harga', 'name': 'Nama_TM'})

    # Usia kosong di FBref diisi dari tanggal lahir Transfermarkt (bukan 0)
    tgl_acuan = df_tm['Tgl_Valuasi'].max()
    usia_tm = ((tgl_acuan - df['date_of_birth']).dt.days / 365.25).round()
    df['Age'] = df['Age'].fillna(usia_tm)

    batas_valuasi = tgl_acuan - pd.Timedelta(days=MAKS_UMUR_VALUASI_HARI)
    basi = df['Tgl_Valuasi'] < batas_valuasi
    print(f"Dibuang {basi.sum()} pemain dengan valuasi sebelum {batas_valuasi.date()}.")
    df = df[~basi & (df['Harga'] > 0)]

    df = df[df['Min'] >= MIN_MENIT].copy()
    df = hitung_fitur(df)
    df = df.drop(columns=['date_of_birth'])

    df = tambah_label_peran(df)
    df, metrik = tambah_prediksi_nilai(df)

    df.to_csv('data/dataset_final_clean.csv', index=False)
    with open('data/model_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrik, f, indent=2)

    print(f"Sukses! Total {len(df)} pemain diproses.")
    print(f"Model valuasi (cross-validation): R² log-harga {metrik['r2_log']:.2f}, "
          f"median error {metrik['median_ape'] * 100:.0f}%, MAE €{metrik['mae_eur']:,.0f}")


if __name__ == "__main__":
    process_data()
