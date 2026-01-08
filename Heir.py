import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
import plotly.express as px
import plotly.graph_objects as go

# 🌙 DARK THEME CSS (Replace light theme)
st.markdown("""
<style>
    /* Dark background */
    .main {background-color: #0e1117;}
    .block-container {padding-top: 2rem;}
    
    /* Sidebar dark */
    .css-1d391kg {background-color: #1a1d24;}
    section[data-testid="stSidebar"] {background-color: #1a1d24;}
    
    /* Text & Headers */
    h1, h2, h3, h4, h5, h6 {color: #ffffff !important;}
    .stMarkdown {color: #e8e8e8;}
    
    /* Metrics - Neon glow */
    .stMetric > label {color: #b8bcc5;}
    .stMetric > .stMetricValue {color: #00d4aa; font-weight: bold;}
    .stMetric {background-color: #1f2937; border: 1px solid #374151;}
    
    /* Dataframe dark */
    .dataframe {background-color: #1a1d24;}
    .dataframe th {background-color: #2a2e3b; color: #ffffff;}
    .dataframe td {background-color: #1a1d24; color: #e8e8e8; border-color: #374151;}
    
    /* Selectbox/Slider dark */
    .stSelectbox div, .stSlider div {background-color: #1a1d24; color: #ffffff;}
    
    /* Checkbox/Buttons */
    .stCheckbox > label {color: #e8e8e8;}
    .stButton > button {background: linear-gradient(45deg, #1f77b4, #00d4aa); 
                       color: white; border-radius: 8px; border: none;}
    
    /* Success/Info boxes */
    div.stAlert {background-color: #1e3a2f; color: #d1fae5;}
    div.element-container .stInfo {background-color: #1e40af; color: #dbeafe;}
    
    /* Plot containers */
    .stPlotlyChart {background-color: #1a1d24;}
</style>
""", unsafe_allow_html=True)

# Title 
st.title("🔗 Hierarchical Clustering Dashboard")

# Sidebar dataset
st.sidebar.header("📊 Dataset")
dataset_choice = st.sidebar.selectbox("Choose Dataset", ["Iris", "Upload CSV"])

if dataset_choice == "Iris":
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    st.success("✅ Iris dataset loaded (150 samples)")
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Custom dataset loaded ({len(df)} samples)")
    else:
        df = pd.DataFrame(load_iris().data, columns=load_iris().feature_names)

# Dataset metrics (dark style)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Rows", len(df), delta_color="normal")
col2.metric("Columns", len(df.columns), delta_color="normal")
col3.metric("Missing Values", df.isnull().sum().sum(), delta_color="inverse")
col4.metric("Memory (MB)", f"{round(df.memory_usage(deep=True).sum()/1024**2, 2)}")

st.subheader("📈 Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

# Clustering controls
st.sidebar.header("⚙️ Clustering Parameters")
linkage_method = st.sidebar.selectbox("Linkage Method", ["ward", "complete", "average", "single"])
n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 3)
distance_metric = st.sidebar.selectbox("Distance Metric", ["euclidean", "manhattan", "cosine"])

scale_features = st.sidebar.checkbox("🔄 Auto-scale Features", value=True)
if scale_features:
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    data = df_scaled
    st.info("✅ Features standardized (zero mean, unit variance)")
else:
    data = df

# Clustering
@st.cache_data
def cluster_data(data, n_clusters, linkage_method, distance_metric):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method, metric=distance_metric)
    return model.fit_predict(data)

labels = cluster_data(data, n_clusters, linkage_method, distance_metric)

# Metrics row
col1, col2, col3 = st.columns(3)
col1.metric("Clusters", len(np.unique(labels)))
col2.metric("Largest Cluster", int(np.bincount(labels).max()))
col3.metric("Silhouette Range", f"{np.min(labels)} to {np.max(labels)}")

# 3D Plotly (dark optimized)
st.subheader("🌐 Interactive 3D Clusters")
if len(df.columns) >= 3:
    fig_3d = px.scatter_3d(
        pd.DataFrame(data, columns=df.columns).iloc[:, :3],
        x=df.columns[0], y=df.columns[1], z=df.columns[2],
        color=labels,
        color_continuous_scale="Viridis",
        title="3D Hierarchical Clusters",
        template="plotly_dark"
    )
    fig_3d.update_layout(paper_bgcolor="#1a1d24", plot_bgcolor="#1a1d24")
    st.plotly_chart(fig_3d, use_container_width=True)
else:
    st.warning("📏 Need 3+ features for 3D plot")

# Enhanced 2D
st.subheader("🎨 2D Clusters")
fig, ax = plt.subplots(figsize=(12, 8), facecolor='#1a1d24')
scatter = ax.scatter(data.iloc[:, 0], data.iloc[:, 1], c=labels, cmap='tab10', 
                    s=80, alpha=0.8, edgecolors='white', linewidth=0.5)
ax.set_facecolor('#0e1117')
ax.set_xlabel(df.columns[0], fontsize=12, color='white')
ax.set_ylabel(df.columns[1], fontsize=12, color='white')
ax.set_title(f"Hierarchical Clustering (n={n_clusters}, {linkage_method})", 
             fontsize=16, color='white', pad=20)
ax.grid(True, alpha=0.3, color='gray')
plt.colorbar(scatter, label="Cluster", shrink=0.8)
st.pyplot(fig)

# Cluster bar
st.subheader("📊 Cluster Distribution")
cluster_df = pd.DataFrame({'Cluster': labels, 'Size': 1}).groupby('Cluster').sum()
st.bar_chart(cluster_df, use_container_width=True)

# Dendrogram
st.subheader("🌳 Dendrogram")
fig_dendro, ax_dendro = plt.subplots(figsize=(14, 6), facecolor='#1a1d24')
linked = linkage(data, method=linkage_method)
dendrogram(linked, truncate_mode="level", p=5, ax=ax_dendro, leaf_rotation=90., 
           above_threshold_color='cyan')
ax_dendro.set_facecolor('#0e1117')
ax_dendro.set_title(f"Dendrogram ({linkage_method} linkage)", fontsize=14, color='white')
ax_dendro.axhline(y=n_clusters*0.8, color='#00d4aa', linestyle='--', linewidth=2, 
                  label=f'Cut: {n_clusters} clusters')
ax_dendro.legend()
plt.tight_layout()
st.pyplot(fig_dendro)

# Download
st.subheader("💾 Export Results")
df_results = df.copy()
df_results['Cluster'] = labels
csv = df_results.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Clustered CSV",
    data=csv,
    file_name=f"hierarchical_clusters_{n_clusters}_clusters.csv",
    mime="text/csv"
)

# Footer
st.markdown("---")
st.markdown("*Made with ❤️ for data clustering | Dark Theme Edition 🌙*")
