import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from io import BytesIO
import matplotlib.pyplot as plt

st.set_page_config(
        page_title="Aplikasi Pertanian",
        page_icon=":bar_chart:",
        layout="wide",
        initial_sidebar_state="auto",
    )

# Fungsi untuk menginisialisasi session state
def init_session():
    return {}

# Fungsi untuk mendapatkan atau membuat session state
def get_session_state():
    session_state = st.session_state.get('_session_state')
    if session_state is None:
        session_state = init_session()
        st.session_state['_session_state'] = session_state
    return session_state

# Fungsi untuk menyimpan data yang diunggah ke session state
def save_uploaded_data(df):
    session_state = get_session_state()
    session_state['uploaded_data'] = df

# Fungsi untuk mendapatkan data yang diunggah dari session state
def get_uploaded_data():
    session_state = get_session_state()
    return session_state.get('uploaded_data', None)

# Fungsi untuk menyimpan pilihan kolom ke dalam session state
def save_selected_columns(selected_columns):
    session_state = get_session_state()
    session_state['selected_columns'] = selected_columns

# Fungsi untuk mendapatkan pilihan kolom dari session state
def get_selected_columns():
    session_state = get_session_state()
    return session_state.get('selected_columns', [])

# Fungsi untuk Unggah file excel
def read_data():
    upload_file = st.file_uploader("ungah file excel",type=["xlsx","xls"])
    if upload_file is not None:
        df=pd.read_excel(upload_file)
        save_uploaded_data(df)
        st.success("Data berhasil di unggah!")
    if st.button("Tampilkan data"):
        upload_data = get_uploaded_data()
        if upload_data is not None:
            st.dataframe(upload_data)
        else:
            st.warning("Data tidak di temukan!")

#Halaman Normalisasi

#fungsi nyimpan hasil label encoding

# Fungsi untuk menyimpan data hasil label encoding ke dalam session state
def save_label_encoded_data(df_encoded):
    session_state = get_session_state()
    session_state['df_encoded'] = df_encoded

# Fungsi untuk mendapatkan data hasil label encoding dari session state
def get_label_encoded_data():
    session_state = get_session_state()
    return session_state.get('df_encoded', None)

#fungsi label encoding
def label_encoding(df, selected_columns):
    df_encoded = df.copy()
    le = LabelEncoder()

    for column in selected_columns:
        if column in df_encoded.columns:
            df_encoded[column] = le.fit_transform(df_encoded[column])

    st.session_state['df_encoded'] = df_encoded
    return df_encoded

# Fungsi untuk menyimpan data hasil missing values ke dalam session state
def save_missing_values_data(df_filled):
    session_state = get_session_state()
    session_state['df_filled'] = df_filled

# Fungsi untuk mendapatkan data hasil missing values dari session state
def get_missing_values_data():
    session_state = get_session_state()
    return session_state.get('df_filled', None)

#fungsi missing values (mean)
def missing_values_mean(df, selected_columns):
    df_filled = df.copy()
    for selected_column in selected_columns:
        if selected_column in df_filled.columns:
            column_mean = df_filled[selected_column].mean()
            df_filled[selected_column].fillna(column_mean, inplace=True)
    
    st.write("Data setelah Missing Values diisi dengan Mean:")
    st.dataframe(df_filled)

    # Menyimpan data hasil missing values ke dalam session state
    save_missing_values_data(df_filled)
    return df_filled

# Fungsi untuk menyimpan data hasil metode MinMax ke dalam session state
def save_minmax_data(df_normalized):
    session_state = get_session_state()
    session_state['df_normalized'] = df_normalized

# Fungsi untuk mendapatkan data hasil metode MinMax dari session state
def get_minmax_data():
    session_state = get_session_state()
    return session_state.get('df_normalized', None)

# Fungsi untuk metode Min-Max Normalisasi
def minmax_normalization(df, selected_columns):
    df_normalized = df.copy()
    min_max_scaler = MinMaxScaler()

    # Lakukan Min-Max Normalisasi 
    df_normalized[selected_columns] = min_max_scaler.fit_transform(df_normalized[selected_columns])

    st.write("Data setelah Min-Max Normalisasi:")
    st.dataframe(df_normalized)

    save_minmax_data(df_normalized)
    return df_normalized

# Fungsi untuk menyimpan data hasil k-means ke dalam session state
def save_kmeans_data(df_clustering_kmeans):
    session_state = get_session_state()
    session_state['df_clustering_kmeans'] = df_clustering_kmeans

# Fungsi untuk mendapatkan data hasil k-means dari session state
def get_kmeans_data():
    session_state = get_session_state()
    return session_state.get('df_clustering_kmeans', None)

# Fungsi untuk K-Means Clustering
def kmeans_clustering(df, selected_columns, num_clusters=3):
    df_clustering_kmeans = df[selected_columns].copy()
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    df_clustering_kmeans['Cluster'] = kmeans.fit_predict(df_clustering_kmeans)
    df_clustering_kmeans['Cluster'] += 1

    # Menyimpan pilihan kolom dan hasil clustering ke dalam session state
    save_selected_columns(selected_columns)
    save_kmeans_data(df_clustering_kmeans)

    return df_clustering_kmeans

# Fungsi untuk menghitung SSE K-Means
def calculate_sse_kmeans(df, selected_columns):
    kmeans = KMeans(n_clusters=len(df['Cluster'].unique()), random_state=0)
    kmeans.fit(df[selected_columns])
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    sse = 0
    for i in range(len(centers)):
        cluster_points = df[labels == i][selected_columns]
        sse += ((cluster_points - centers[i]) ** 2).sum().sum()
    return sse

# Fungsi untuk menyimpan data hasil k-means ke dalam session state
def save_kmedoids_data(df_clustering_kmedoids):
    session_state = get_session_state()
    session_state['df_clustering_kmedoids'] = df_clustering_kmedoids

# Fungsi untuk mendapatkan data hasil k-means dari session state
def get_kmedoids_data():
    session_state = get_session_state()
    return session_state.get('df_clustering_kmedoids', None)

# Fungsi untuk K-Medoids Clustering
def kmedoids_clustering(df, selected_columns, num_clusters=3):
    df_clustering_kmedoids = df[selected_columns].copy()
    kmedoids = KMedoids(n_clusters=num_clusters, random_state=0)
    df_clustering_kmedoids['Cluster'] = kmedoids.fit_predict(df_clustering_kmedoids)
    df_clustering_kmedoids['Cluster'] += 1

    # Menyimpan pilihan kolom dan hasil clustering ke dalam session state
    save_selected_columns(selected_columns)
    save_kmedoids_data(df_clustering_kmedoids)

    return df_clustering_kmedoids

# Fungsi untuk menghitung SSE
def calculate_sse_kmedoids(df, selected_columns):
    kmedoids = KMedoids(n_clusters=len(df['Cluster'].unique()), random_state=0)
    kmedoids.fit(df[selected_columns])
    centers = kmedoids.cluster_centers_
    labels = kmedoids.labels_
    sse = 0
    for i in range(len(centers)):
        cluster_points = df[labels == i][selected_columns]
        sse += ((cluster_points - centers[i]) ** 2).sum().sum()
    return sse

#fungsi untuk k-means hasilnya kelompok desa
def kmeans_tabel_clusters(data_kmeans, data, num_clusters):
    for cluster_num in range(1, num_clusters + 1):
        st.write(f"**Cluster {cluster_num}**")
        result_data_kmeans = data_kmeans[data_kmeans['Cluster'] == cluster_num][["Luas Tanaman (Ha)", "Varietas/jenis", "Stadia/Umur Tanaman (hst)", 
                    "Jenis OPT", "Pupuk Bersubsidi Organik dan Anorganik (Ton)", "Luas Terserang (Ha)", "Intensitas (%)",
                    "Luas Waspada (Ha)", "Hasil Panen (Ton)"]]

        # Menambahkan kolom "Desa" dari variabel data
        result_data_kmeans["Desa"] = data[data_kmeans['Cluster'] == cluster_num]["Desa"].values

        # Mengubah urutan kolom sehingga "Desa" berada di awal
        result_data_kmeans = result_data_kmeans[['Desa'] + [col for col in result_data_kmeans if col != 'Desa']]
        
        # Hitung jumlah desa dalam cluster
        num_villages_in_cluster = len(result_data_kmeans)
        
        # Tampilkan jumlah desa di bawah tabel
        st.write(f"Jumlah Desa dalam Cluster {cluster_num}: {num_villages_in_cluster}")
        
        st.dataframe(result_data_kmeans)

def kmedoids_tabel_clusters(data_kmedoids, data, num_clusters):
    for cluster_num in range(1, num_clusters + 1):
        st.write(f"**Cluster {cluster_num}**")
        result_data_kmedoids = data_kmedoids[data_kmedoids['Cluster'] == cluster_num][["Luas Tanaman (Ha)", "Varietas/jenis", "Stadia/Umur Tanaman (hst)", 
                    "Jenis OPT", "Pupuk Bersubsidi Organik dan Anorganik (Ton)", "Luas Terserang (Ha)", "Intensitas (%)",
                    "Luas Waspada (Ha)", "Hasil Panen (Ton)"]]

        # Menambahkan kolom "Desa" dari variabel data
        result_data_kmedoids["Desa"] = data[data_kmedoids['Cluster'] == cluster_num]["Desa"].values

        # Mengubah urutan kolom sehingga "Desa" berada di awal
        result_data_kmedoids = result_data_kmedoids[['Desa'] + [col for col in result_data_kmedoids if col != 'Desa']]
        
        # Hitung jumlah desa dalam cluster
        num_villages_in_cluster = len(result_data_kmedoids)
        
        # Tampilkan jumlah desa di bawah tabel
        st.write(f"Jumlah Desa dalam Cluster {cluster_num}: {num_villages_in_cluster}")
        
        st.dataframe(result_data_kmedoids)

#fungsi presentase
# Fungsi untuk menghitung dan menampilkan presentase setiap kolom untuk setiap cluster
def show_cluster_column_percentages(data_clustered, num_clusters):
    # Kolom numerik yang ingin Anda hitung persentase tertingginya
    kolom_numerik = ["Luas Tanaman (Ha)", "Varietas/jenis", "Stadia/Umur Tanaman (hst)", 
                        "Jenis OPT", "Pupuk Bersubsidi Organik dan Anorganik (Ton)", "Luas Terserang (Ha)", "Intensitas (%)",
                        "Luas Waspada (Ha)", "Hasil Panen (Ton)"]

    # Inisialisasi dictionary untuk menyimpan hasil presentase tertinggi untuk setiap cluster
    presentase_tertinggi = {cluster: {} for cluster in range(1, num_clusters + 1)}

    # Iterasi melalui setiap cluster
    for cluster in range(1, num_clusters + 1):
        presentase_tertinggi[cluster] = {}

        # Filter data berdasarkan cluster
        data_cluster = data_clustered[data_clustered['Cluster'] == cluster]

        for kolom in kolom_numerik:
            #nilai_tertinggi = data_cluster[kolom].max()
            total_nilai = data_cluster[kolom].sum()
            
            presentase = (total_nilai / 690) * 100
            presentase_tertinggi[cluster][kolom] = f"{presentase:.2f}%"

    # Menyusun data presentase dalam bentuk DataFrame
    df_presentase = pd.DataFrame.from_dict(presentase_tertinggi, orient='index')
    df_presentase.index.name = 'Cluster'
    
    return df_presentase

#Fungsi grafik SSE k-Means dan k-medoids
def calculate_sse_for_multiple_clusters_kmeans(df, selected_columns):
    sse_values = []
    for num_clusters in range(2, 11):  # Hitung SSE untuk kluster dari 2 hingga 10
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        kmeans.fit(df[selected_columns])
        sse = kmeans.inertia_
        sse_values.append(sse)
    return sse_values

def calculate_sse_for_multiple_clusters_kmedoids(df, selected_columns):
    sse_values = []
    for num_clusters in range(2, 11):  # Hitung SSE untuk kluster dari 2 hingga 10
        kmedoids = KMedoids(n_clusters=num_clusters, random_state=0)
        kmedoids.fit(df[selected_columns])
        centers = df.iloc[kmedoids.medoid_indices_][selected_columns].values
        labels = kmedoids.labels_
        sse = 0
        for i in range(len(centers)):
            cluster_points = df[labels == i][selected_columns].values
            sse += ((cluster_points - centers[i]) ** 2).sum()
        sse_values.append(sse)
    return sse_values


def main():

    #pilih  menu
    menu=("Aplikasi CLustering",["Beranda","Normalisasi","K-Means","K-Medoids","Hasil Clustering",
                                 "Analisa setiap Cluster"])

    # Tampilkan judul "Aplikasi Pertanian" di bagian atas kiri
    st.sidebar.title(menu[0])

    #tampilan menu
    select_menu = st.sidebar.selectbox("Pilih Menu : ", menu[1])

    if select_menu == "Beranda":
        st.header("Program K-Means dan K-Medoids serta SSE")
        st.subheader("Halaman Beranda")
        df = read_data()
        return df
    
    

    elif select_menu == "Normalisasi":
        submenu_normalisasi = ["Label Encoding","Missing Values (Mean)","Normalisasi MinMax"]
        select_submenu_normalisasi = st.sidebar.selectbox("Pilih SubMenu : ", submenu_normalisasi)
        if select_submenu_normalisasi == "Label Encoding":
            st.subheader("Halaman Label Encoding")
            uploaded_data = get_uploaded_data()

            if uploaded_data is not None:
                selected_columns = ['Varietas/jenis','Jenis OPT']
                if selected_columns:
                    df_encoded = label_encoding(uploaded_data, selected_columns)

                    # Buat daftar kolom yang ingin Anda tampilkan
                    columns_to_display = ["Luas Tanaman (Ha)", "Varietas/jenis", "Stadia/Umur Tanaman (hst)", 
                    "Jenis OPT", "Pupuk Bersubsidi Organik dan Anorganik (Ton)", "Luas Terserang (Ha)", "Intensitas (%)",
                    "Luas Waspada (Ha)", "Hasil Panen (Ton)"]

                    st.write("Data setelah Label Encoding:")
                    st.dataframe(df_encoded[columns_to_display])

                    # Menyimpan data hasil label encoding ke dalam session state
                    save_label_encoded_data(df_encoded)
            else:
                st.warning("belum ada data yang di unggah")

        elif select_submenu_normalisasi == "Missing Values (Mean)":
            st.subheader("Halaman Imputasi Data (Mean)")
            df_encoded = get_label_encoded_data()  # Mendapatkan data hasil label encoding dari session state
            
            if df_encoded is not None:
                selected_columns = ["Luas Tanaman (Ha)", "Varietas/jenis", "Stadia/Umur Tanaman (hst)", 
                "Jenis OPT", "Pupuk Bersubsidi Organik dan Anorganik (Ton)", "Luas Terserang (Ha)", "Intensitas (%)",
                "Luas Waspada (Ha)", "Hasil Panen (Ton)"]
                if selected_columns:

                    # Buat daftar kolom yang ingin Anda tampilkan
                    columns_to_display = ["Luas Tanaman (Ha)", "Varietas/jenis", "Stadia/Umur Tanaman (hst)", 
                    "Jenis OPT", "Pupuk Bersubsidi Organik dan Anorganik (Ton)", "Luas Terserang (Ha)", "Intensitas (%)",
                    "Luas Waspada (Ha)", "Hasil Panen (Ton)"]

                    missing_values_mean(df_encoded[columns_to_display], selected_columns)
            else:
                st.warning("belum ada data yang di unggah")
            

        elif select_submenu_normalisasi == "Normalisasi MinMax":
            st.subheader("Halaman Normalisasi MinMax")
            df_filled = get_missing_values_data()  # Dapatkan data hasil "Missing Values (Mean)" dari session state
            if df_filled is not None:
                selected_columns = ["Luas Tanaman (Ha)", "Varietas/jenis", "Stadia/Umur Tanaman (hst)", 
                "Jenis OPT", "Pupuk Bersubsidi Organik dan Anorganik (Ton)", "Luas Terserang (Ha)", "Intensitas (%)",
                "Luas Waspada (Ha)", "Hasil Panen (Ton)"]
                if selected_columns:

                    # Buat daftar kolom yang ingin Anda tampilkan
                    columns_to_display = ["Luas Tanaman (Ha)", "Varietas/jenis", "Stadia/Umur Tanaman (hst)", 
                    "Jenis OPT", "Pupuk Bersubsidi Organik dan Anorganik (Ton)", "Luas Terserang (Ha)", "Intensitas (%)",
                    "Luas Waspada (Ha)", "Hasil Panen (Ton)"]

                    minmax_normalization(df_filled[columns_to_display], selected_columns)
            else:
                st.warning("belum ada data yang di unggah")
                    

    elif select_menu == "K-Means":
        st.header("Halaman K-Means dan SSE")
        df_normalized = get_minmax_data()
        
        if df_normalized is not None:
            selected_columns = ["Luas Tanaman (Ha)", "Varietas/jenis", "Stadia/Umur Tanaman (hst)", 
                "Jenis OPT", "Pupuk Bersubsidi Organik dan Anorganik (Ton)", "Luas Terserang (Ha)", "Intensitas (%)",
                "Luas Waspada (Ha)", "Hasil Panen (Ton)"]
            if selected_columns:
                num_clusters = st.number_input('Masukkan jumlah kluster:', min_value=1, value=3) # Jumlah kluster yang diinginkan (3 kluster)
                
                # Simpan num_clusters ke dalam session state
                st.session_state['num_clusters'] = num_clusters

                df_clustering_kmeans = kmeans_clustering(df_normalized, selected_columns, num_clusters)
                st.write("Data setelah K-Means Clustering:")
                st.dataframe(df_clustering_kmeans)

                # Menghitung SSE
                sse = calculate_sse_kmeans(df_clustering_kmeans, selected_columns)
                st.write("SSE (Sum of Squared Errors):", sse)
                st.write("")

                if st.button("Tampilkan Grafik SSE K-Means"):
                    # Menghitung SSE untuk K-Means
                    sse_values_kmeans = calculate_sse_for_multiple_clusters_kmeans(df_clustering_kmeans, selected_columns)
                    
                    # Menampilkan grafik SSE untuk K-Means
                    st.write("SSE Graph for K-Means")
                    plt.figure(figsize=(10, 5))
                    plt.plot(range(2, 11), sse_values_kmeans, marker='o', linestyle='-')
                    plt.title('SSE vs Number of Clusters (K-Means)')
                    plt.xlabel('Number of Clusters')
                    plt.ylabel('SSE')
                    st.pyplot(plt)
        else:
            st.warning("belum ada data yang di unggah")

    elif select_menu == "K-Medoids":
        st.subheader("Halaman K-Medoids dan SSE")
        df_normalized = get_minmax_data()
        
        if df_normalized is not None:
            selected_columns = ["Luas Tanaman (Ha)", "Varietas/jenis", "Stadia/Umur Tanaman (hst)", 
                "Jenis OPT", "Pupuk Bersubsidi Organik dan Anorganik (Ton)", "Luas Terserang (Ha)", "Intensitas (%)",
                "Luas Waspada (Ha)", "Hasil Panen (Ton)"]
            if selected_columns:
                num_clusters = st.number_input('Masukkan jumlah kluster:', min_value=1, value=3)  # Jumlah kluster yang diinginkan (3 kluster)

                # Simpan num_clusters ke dalam session state
                st.session_state['num_clusters'] = num_clusters

                df_clustering_kmedoids = kmedoids_clustering(df_normalized, selected_columns, num_clusters)
                st.write("Data setelah K-Medoids Clustering:")
                st.dataframe(df_clustering_kmedoids)

                # Menghitung SSE
                sse = calculate_sse_kmedoids(df_clustering_kmedoids, selected_columns)
                st.write("SSE (Sum of Squared Errors):", sse)
                st.write("")

                if st.button("Tampilkan Grafik SSE K-Medoids"):
                    # Menghitung SSE untuk K-Medoids
                    sse_values_kmedoids = calculate_sse_for_multiple_clusters_kmedoids(df_clustering_kmedoids, selected_columns)
                    
                    # Menampilkan grafik SSE untuk K-Medoids
                    st.write("SSE Graph for K-Means")
                    plt.figure(figsize=(10, 5))
                    plt.plot(range(2, 11), sse_values_kmedoids, marker='o', linestyle='-')
                    plt.title('SSE vs Number of Clusters (K-Medoids)')
                    plt.xlabel('Number of Clusters')
                    plt.ylabel('SSE')
                    st.pyplot(plt)
        else:
            st.warning("belum ada data yang di unggah")
        
    elif select_menu == "Hasil Clustering":
        submenu_clustering = st.sidebar.selectbox("Pilih Metode Clustering:", ["Hasil K-Means", "Hasil K-Medoids"])
        st.subheader("Halaman Hasil Clustering")
        data = get_uploaded_data()

        # Mengambil num_clusters dari session state
        num_clusters = st.session_state.get('num_clusters', 3)

        if data is not None:
            if submenu_clustering == "Hasil K-Means":
                data_kmeans = get_kmeans_data()
                if data_kmeans is not None:
                    st.write("**Jumlah Desa hasil K-Means Clustering**")
                    kmeans_tabel_clusters(data_kmeans, data, num_clusters)
                    st.write("")
            elif submenu_clustering == "Hasil K-Medoids":
                data_kmedoids = get_kmedoids_data()
                if data_kmedoids is not None:
                    st.write("**Jumlah Desa hasil K-Medoids Clustering**")
                    kmedoids_tabel_clusters(data_kmedoids, data, num_clusters)
                    st.write("")
        else:
            st.warning("Data tidak ada!")

    elif select_menu == "Analisa setiap Cluster":
        submenu_presentase = st.sidebar.selectbox("Pilih Metode Clustering:", ["Analisa Cluster pada K-Means", "Analisa Cluster pada K-Medoids"])
        st.subheader("Halaman Analisa setiap Cluster")
        data = get_uploaded_data()

        if data is not None:
            if submenu_presentase == "Analisa Cluster pada K-Means":
                data_kmeans = get_kmeans_data()
                if data_kmeans is not None:
                    st.write("**Analisa K-Means Clustering**")
                    num_clusters_kmeans = len(data_kmeans['Cluster'].unique())
                    df_kmeans_presentase = show_cluster_column_percentages(data_kmeans, num_clusters_kmeans)  # Dapatkan presentase di setiap kolom dari data K-Means
                    st.write("Presentase dari metode K-Means")
                    st.dataframe(df_kmeans_presentase)

            elif submenu_presentase == "Analisa Cluster pada K-Medoids":
                data_kmedoids = get_kmedoids_data()
                if data_kmedoids is not None:
                    st.write("**Analisa K-Medoids Clustering**")
                    num_clusters_kmedoids = len(data_kmedoids['Cluster'].unique())
                    df_kmedoids_presentase = show_cluster_column_percentages(data_kmedoids, num_clusters_kmedoids)  # Dapatkan presentase di setiap kolom dari data K-Medoids
                    st.write("Presentase dari metode K-Medoids")
                    st.dataframe(df_kmedoids_presentase)
        else:
            st.warning("Data tidak ada!")





if __name__ == "__main__":
    main()