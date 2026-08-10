import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import os

def process_data():
    print("Memulai proses pembersihan dan penggabungan dataset...")

    # Konfigurasi file FBref
    fbref_files = {
        'data/data_fbref_standard.csv': 'Premier League',
        'data/data_fbref_serieA.csv': 'Serie A',
        'data/data_fbref_eredivisie.csv': 'Eredivisie',
        'data/data_fbref_belgian.csv': 'Belgian Pro League',
        'data/data_fbref_J1.csv': 'J1 League',
        'data/data_fbref_aus.csv': 'A-League',
        'data/data_fbref_portugal.csv': 'Liga Portugal',
        'data/data_fbref_mls.csv': 'MLS',
        'data/data_fbref_laliga.csv': 'La Liga',
    }

    # 1. Load FBref
    dfs = []
    for file_path, league in fbref_files.items():
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, header=1)
            df.columns = df.columns.str.strip()
            df['Asal_Liga'] = league
            dfs.append(df)

    if not dfs:
        raise ValueError("Data FBref tidak ditemukan!")

    df_fbref = pd.concat(dfs, ignore_index=True)
    df_fbref.rename(columns={'Player': 'Nama', 'player': 'Nama'}, inplace=True, errors='ignore')

    # 2. Load Market Data (Kaggle)
    print("Memproses data pasar Transfermarkt...")
    df_val = pd.read_csv('data/player_valuations.csv')
    df_play = pd.read_csv('data/players.csv')
    
    # Merge Valuasi + Players
    df_market = pd.merge(df_val, df_play[['player_id', 'name']], on='player_id', how='inner')
    df_market.rename(columns={'name': 'Nama', 'market_value_in_eur': 'Harga'}, inplace=True)
    df_market = df_market.sort_values('date').drop_duplicates(subset=['Nama'], keep='last')

    # 3. Merge FBref + Market
    df_final = pd.merge(df_fbref, df_market[['Nama', 'Harga']], on='Nama', how='inner')

    # 4. Cleaning & P90 Calculation
    df_final['Min'] = pd.to_numeric(df_final['Min'].astype(str).str.replace(',', ''), errors='coerce')
    df_clean = df_final[df_final['Min'] >= 900].copy()
    
    for col in ['Gls', 'Ast']:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)

    df_clean['Gls_p90'] = df_clean['Gls'] / pd.to_numeric(df_clean['90s'])
    df_clean['Ast_p90'] = df_clean['Ast'] / pd.to_numeric(df_clean['90s'])
    df_clean.fillna(0, inplace=True)

    # 5. Scaling & Clustering (K-Means)
    scaler = MinMaxScaler()
    fitur = ['Gls_p90', 'Ast_p90']
    df_clean[fitur] = scaler.fit_transform(df_clean[fitur])

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df_clean['Cluster'] = kmeans.fit_predict(df_clean[fitur])

    # Simpan hasil
    df_clean.to_csv('data/dataset_final_clean.csv', index=False)
    print(f"Sukses! Total {len(df_clean)} pemain diproses.")

if __name__ == "__main__":
    process_data()