import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.neighbors import NearestNeighbors

# Konfigurasi halaman web
st.set_page_config(
    page_title="Moneyball Player Scouting System",
    layout="wide"
)

st.title("Moneyball Player Scouting System")
st.markdown("Sistem analisis dan rekomendasi alternatif pemain berbasis statistik performa dan valuasi pasar.")

# Memuat data dengan cache
@st.cache_data
def load_data():
    return pd.read_csv('data/dataset_final_clean.csv')

df = load_data()

# Panel Sidebar untuk Input Pengguna
st.sidebar.header("Parameter Pencarian")
all_players = df['Nama'].unique()
target_player = st.sidebar.selectbox("Pilih Pemain Target:", all_players)
max_age = st.sidebar.slider("Batas Maksimal Usia", min_value=16, max_value=40, value=30)
filter_pos = st.sidebar.checkbox("Gunakan Filter Posisi yang Sama", value=True)

# Fitur model untuk KNN & K-Means
fitur_model = ['Gls_p90', 'Ast_p90']

# Eksekusi Analisis ketika tombol ditekan
if st.sidebar.button("Jalankan Analisis", type="primary"):
    match_target = df[df['Nama'] == target_player]
    
    if match_target.empty:
        st.error("Data pemain target tidak ditemukan.")
    else:
        target_idx = match_target.index[0]
        target_row = df.iloc[target_idx]
        
        # Ambil posisi utama target
        pos_utama = str(target_row['Pos']).split(',')[0].strip() if 'Pos' in df.columns else ""
        
        # Filter dataset berdasarkan posisi dan batas usia
        if filter_pos and pos_utama:
            df_pool = df[df['Pos'].str.contains(pos_utama, na=False)].copy()
        else:
            df_pool = df.copy()
            
        df_pool = df_pool[df_pool['Age'] <= max_age]
        
        if len(df_pool) < 2:
            st.warning("Data terlalu sedikit dengan filter posisi tersebut. Coba matikan 'Filter Posisi' atau naikkan batas usia di sidebar.")
        else:
            # Setup model KNN berdasarkan pool data yang sudah difilter
            X_pool = df_pool[fitur_model].values
            knn = NearestNeighbors(n_neighbors=min(11, len(df_pool)), metric='euclidean')
            knn.fit(X_pool)
            
            if target_player in df_pool['Nama'].values:
                pool_target_idx = df_pool[df_pool['Nama'] == target_player].index[0]
                target_vector = df_pool.loc[[pool_target_idx], fitur_model].values
                
                distances, indices = knn.kneighbors(target_vector)
                
                # Tampilkan profil target (3 kolom)
                st.markdown(f"### Profil Target: {target_row['Nama']}")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Klub", target_row['Squad'])
                with col2:
                    st.metric("Liga", target_row['Asal_Liga'])
                with col3:
                    st.metric("Valuasi Pasar", f"€{target_row['Harga']:,.0f}")
                
                results = []
                for i in range(1, len(indices[0])): 
                    idx = indices[0][i]
                    p = df_pool.iloc[indices[0][i]]
                    sim = round((1 / (1 + distances[0][i])) * 100, 2)
                    price = p['Harga']
                    age_val = p['Age'] if 'Age' in p else 25
                    
                    roi = round(sim / (price / 1_000_000), 2) if price > 0 else 0
                    
                    # Pastikan kolom Cluster aman dibaca
                    cluster_val = p['Cluster'] if 'Cluster' in p else 0
                    
                    results.append({
                        "Pemain": p['Nama'],
                        "Klub": p['Squad'],
                        "Liga": p['Asal_Liga'],
                        "Usia": age_val,
                        "Kluster Peran": f"Kluster {cluster_val}",
                        "Kemiripan (%)": sim,
                        "Harga": f"€{price:,.0f}",
                        "Moneyball ROI": roi
                    })
                
                if results:
                    results_df = pd.DataFrame(results).sort_values("Moneyball ROI", ascending=False)
                    results_df = results_df.reset_index(drop=True)
                    results_df.index = results_df.index + 1
                    results_df.index.name = "No"
                    
                    st.write(f"### Daftar Rekomendasi Pemain Alternatif (Filter Posisi: {pos_utama if filter_pos else 'Bebas'})")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Panduan interpretasi metrik ROI
                    st.info("""
                    **Panduan Interpretasi Skor Moneyball ROI:**
                    * **Nilai Tinggi**: Menunjukkan efisiensi ekonomi yang baik. Pemain memiliki kemiripan statistik yang tinggi dengan valuasi pasar yang relatif rendah (potensi *underpriced*).
                    * **Nilai Rendah**: Menunjukkan efisiensi yang kurang optimal atau valuasi pasar yang relatif tinggi (*overpriced*) meskipun memiliki kemiripan statistik.
                    """)
                    
                    # --- VISUALISASI K-MEANS CLUSTERING (SCATTER PLOT) ---
                    st.write("### Peta Sebaran Kluster Peran Pemain (Model K-Means)")
                    st.markdown("*Grafik ini memetakan posisi pemain berdasarkan output statistik per 90 menit yang dikelompokkan otomatis ke dalam kluster peran taktis.*")
                    
                    # Konversi Cluster jadi string biar warnanya kebaca sebagai kategori di Plotly
                    df_pool['Cluster_Str'] = df_pool['Cluster'].astype(str) if 'Cluster' in df_pool.columns else '0'
                    
                    fig = px.scatter(
                        df_pool, 
                        x='Gls_p90', 
                        y='Ast_p90', 
                        color='Cluster_Str', 
                        hover_data=['Nama', 'Squad', 'Harga', 'Age'],
                        labels={'Gls_p90': 'Gol per 90 Menit', 'Ast_p90': 'Assist per 90 Menit', 'Cluster_Str': 'Kluster Peran'},
                        title="Kluster Peran Pemain Berdasarkan Statistik Performa"
                    )
                    
                    # Beri tanda bintang warna merah menyala buat si Pemain Target Utama
                    fig.add_scatter(
                        x=[target_row['Gls_p90']], 
                        y=[target_row['Ast_p90']], 
                        mode='markers', 
                        marker=dict(size=18, color='red', symbol='star'), 
                        name='Target Utama'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Tidak ditemukan pemain alternatif yang sesuai dengan kriteria.")
            else:
                st.error("Pemain target tidak berada dalam jangkauan filter data saat ini.")