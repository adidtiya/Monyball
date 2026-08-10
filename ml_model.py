import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Load data
df = pd.read_csv('data/dataset_final_clean.csv')

def cari_pemain(nama_target, filter_pos=True):
    match = df[df['Nama'].str.contains(nama_target, case=False, na=False)]
    if match.empty:
        print("Pemain tidak ditemukan.")
        return

    target_idx = match.index[0]
    target_row = df.iloc[target_idx]
    
    # Ambil posisi utama (misal: "MF, FW" menjadi "MF")
    pos_utama = str(target_row['Pos']).split(',')[0].strip()
    
    # Filter dataset jika filter_pos aktif
    df_filtered = df[df['Pos'].str.contains(pos_utama, na=False)] if filter_pos else df
    
    # KNN training
    knn = NearestNeighbors(n_neighbors=min(11, len(df_filtered)), metric='euclidean')
    knn.fit(df_filtered[['Gls_p90', 'Ast_p90']].values)
    
    # Cari tetangga
    dist, indices = knn.kneighbors(df_filtered.loc[[target_idx], ['Gls_p90', 'Ast_p90']].values)
    
    print(f"\nRekomendasi untuk {target_row['Nama']} (Posisi: {pos_utama}):")
    for i in range(1, len(indices[0])):
        p = df_filtered.iloc[indices[0][i]]
        print(f"- {p['Nama']} | Klub: {p['Squad']} | Cluster: {p['Cluster']}")

if __name__ == "__main__":
    cari_pemain("bruno fernandes", filter_pos=True)