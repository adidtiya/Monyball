import numpy as np
import plotly.express as px
import streamlit as st

from ml_model import cari_alternatif, indikasi_nilai, load_dataset

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
    df, metrik = load_dataset()
    # Label unik agar pemain bernama sama tetap bisa dibedakan
    df['Label'] = df['Nama'] + ' (' + df['Squad'] + ')'
    return df, metrik


df, metrik = load_data()
ambang = metrik['ambang_undervalued_pct']

# Panel Sidebar untuk Input Pengguna
st.sidebar.header("Parameter Pencarian")
labels = df.sort_values('Nama')['Label'].tolist()
target_label = st.sidebar.selectbox("Pilih Pemain Target:", labels)
max_age = st.sidebar.slider("Batas Maksimal Usia", min_value=16, max_value=41, value=30)
filter_pos = st.sidebar.checkbox("Gunakan Filter Posisi yang Sama", value=True)
semua_liga = sorted(df['Asal_Liga'].unique())
liga = st.sidebar.multiselect("Liga", semua_liga, default=semua_liga)
budget = st.sidebar.number_input("Budget Maksimal (juta €, 0 = bebas)", min_value=0.0, value=0.0, step=1.0)
jumlah = st.sidebar.slider("Jumlah Rekomendasi", min_value=5, max_value=30, value=10)
urutan = st.sidebar.radio("Urutkan Berdasarkan", ["Kemiripan", "Potensi Undervalued"])

if st.sidebar.button("Jalankan Analisis", type="primary"):
    st.session_state['analisis'] = True

if not st.session_state.get('analisis'):
    st.info("Pilih pemain target di sidebar lalu tekan **Jalankan Analisis**.")
    st.stop()

target_row = df[df['Label'] == target_label].iloc[0]

# Tampilkan profil target
st.markdown(f"### Profil Target: {target_row['Nama']}")
col1, col2, col3 = st.columns(3)
col1.metric("Klub", target_row['Squad'])
col2.metric("Liga", target_row['Asal_Liga'])
col3.metric("Posisi / Usia", f"{target_row['Pos']} / {target_row['Age']:.0f}")
col4, col5, col6 = st.columns(3)
col4.metric("Valuasi Pasar", f"€{target_row['Harga']:,.0f}")
col6.metric("Peran (K-Means)", target_row['Peran'])
col5.metric(
    "Prediksi Model", f"€{target_row['Nilai_Prediksi']:,.0f}",
    f"{target_row['Undervalued_pct']:+.0f}% ({indikasi_nilai(target_row['Undervalued_pct'], ambang)})",
)

if target_row['Pos_Utama'] == 'GK':
    st.warning("Data FBref yang dipakai belum memuat statistik kiper (save%, clean sheet), "
               "jadi kemiripan antar kiper hanya berdasarkan menit & starter, bukan kualitas.")

hasil = cari_alternatif(
    df, target_row['player_id'],
    filter_pos=filter_pos,
    max_age=max_age,
    liga=liga,
    max_harga=budget * 1_000_000 if budget > 0 else None,
    k=jumlah,
)

if hasil.empty:
    st.warning("Tidak ada pemain yang cocok dengan filter. Coba matikan 'Filter Posisi', "
               "tambah liga, naikkan batas usia, atau naikkan budget.")
    st.stop()

kolom_sort = 'Kemiripan' if urutan == "Kemiripan" else 'Undervalued_pct'
hasil = hasil.sort_values(kolom_sort, ascending=False)
tabel = hasil.assign(
    Indikasi=hasil['Undervalued_pct'].map(lambda v: indikasi_nilai(v, ambang)),
    Age=hasil['Age'].astype(int),
)[['Nama', 'Squad', 'Asal_Liga', 'Pos', 'Age', 'Peran', 'npG_p90', 'Ast_p90', 'Kemiripan',
   'Harga', 'Nilai_Prediksi', 'Undervalued_pct', 'Indikasi']]
tabel.index = np.arange(1, len(tabel) + 1)
tabel.index.name = "No"

st.write(f"### Daftar Rekomendasi Pemain Alternatif "
         f"(Filter Posisi: {target_row['Pos_Utama'] if filter_pos else 'Bebas'})")
st.dataframe(
    tabel,
    width="stretch",
    column_config={
        'Nama': "Pemain",
        'Squad': "Klub",
        'Asal_Liga': "Liga",
        'Pos': "Posisi",
        'Age': "Usia",
        'Peran': "Peran (K-Means)",
        'npG_p90': st.column_config.NumberColumn("Gol non-PK/90", format="%.2f"),
        'Ast_p90': st.column_config.NumberColumn("Assist/90", format="%.2f"),
        'Kemiripan': st.column_config.ProgressColumn("Kemiripan", format="%.0f%%", min_value=0, max_value=100),
        'Harga': st.column_config.NumberColumn("Harga Pasar (€)", format="euro"),
        'Nilai_Prediksi': st.column_config.NumberColumn("Prediksi Model (€)", format="euro"),
        'Undervalued_pct': st.column_config.NumberColumn("Potensi Undervalued", format="%+.0f%%"),
    },
)

st.info(f"""
**Cara membaca hasil:**
* **Kemiripan**: 90% berarti gaya main (gol non-penalti/90, assist/90, rasio starter) pemain ini lebih dekat
  ke target daripada 90% pasangan pemain acak di dataset.
* **Prediksi Model**: valuasi wajar dari model Gradient Boosting berdasarkan statistik, usia, menit bermain, liga,
  dan posisi. Akurasi (cross-validation, {metrik['n_pemain']} pemain): R² log-harga **{metrik['r2_log']:.2f}**,
  median error **{metrik['median_ape'] * 100:.0f}%**.
* **Indikasi**: *Undervalued* jika prediksi ≥ {ambang:.0f}% di atas harga pasar, *Overvalued* jika jauh di bawahnya.
  Selisih yang lebih kecil masih dalam batas error model, jadi dianggap *Wajar*.
* Pemain yang sangat murah cenderung tampak undervalued (efek regresi ke rata-rata). Bandingkan pemain dalam liga
  atau budget yang sama, dan cek faktor lain (cedera, kontrak, kualitas bertahan) yang tidak tercakup data.
""")

col_kiri, col_kanan = st.columns(2)

with col_kiri:
    # --- VISUALISASI K-MEANS CLUSTERING (SCATTER PLOT) ---
    st.write("#### Peta Peran Pemain (K-Means)")
    pool_plot = df[df['Pos'].str.contains(target_row['Pos_Utama'], na=False)] if filter_pos else df
    fig = px.scatter(
        pool_plot,
        x='npG_p90',
        y='Ast_p90',
        color='Peran',
        hover_data=['Nama', 'Squad', 'Harga', 'Age'],
        labels={'npG_p90': 'Gol non-penalti per 90 menit', 'Ast_p90': 'Assist per 90 menit'},
        opacity=0.5,
    )
    fig.add_scatter(
        x=hasil['npG_p90'], y=hasil['Ast_p90'], mode='markers', name='Rekomendasi',
        marker=dict(size=11, color='rgba(0,0,0,0)', line=dict(width=2, color='black')),
        text=hasil['Nama'], hoverinfo='text',
    )
    fig.add_scatter(
        x=[target_row['npG_p90']], y=[target_row['Ast_p90']], mode='markers', name='Target Utama',
        marker=dict(size=18, color='red', symbol='star'),
    )
    st.plotly_chart(fig, width="stretch")

with col_kanan:
    st.write("#### Harga Pasar vs Prediksi Model")
    fig2 = px.scatter(
        hasil, x='Harga', y='Nilai_Prediksi', color='Asal_Liga', hover_data=['Nama', 'Squad'],
        log_x=True, log_y=True,
        labels={'Harga': 'Harga pasar (€)', 'Nilai_Prediksi': 'Prediksi model (€)', 'Asal_Liga': 'Liga'},
    )
    batas = [min(hasil['Harga'].min(), hasil['Nilai_Prediksi'].min()),
             max(hasil['Harga'].max(), hasil['Nilai_Prediksi'].max())]
    fig2.add_scatter(x=batas, y=batas, mode='lines', name='Harga = prediksi',
                     line=dict(dash='dash', color='gray'))
    fig2.update_traces(marker=dict(size=11), selector=dict(mode='markers'))
    st.plotly_chart(fig2, width="stretch")
    st.caption("Titik di atas garis putus-putus: model menilai pemain lebih mahal dari harga pasarnya.")
