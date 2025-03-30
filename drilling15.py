import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import io

# Configuration de la page
st.set_page_config(
    page_title="Mining Geology Data Application",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styles CSS personnalisés
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E4053;
        text-align: center;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #3498DB;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #2E4053;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .author {
        font-size: 1rem;
        color: #566573;
        font-style: italic;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #F8F9F9;
        padding: 10px 15px;
        border-radius: 4px 4px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3498DB !important;
        color: white !important;
    }
    .stButton>button {
        background-color: #3498DB;
        color: white;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #2874A6;
        color: white;
    }
    .uploadedFile {
        border: 1px solid #3498DB;
        border-radius: 5px;
        padding: 10px;
    }
    .success-message {
        background-color: #D4EFDF;
        border-left: 5px solid #2ECC71;
        padding: 10px;
        border-radius: 0px 5px 5px 0px;
    }
    .warning-message {
        background-color: #FCF3CF;
        border-left: 5px solid #F1C40F;
        padding: 10px;
        border-radius: 0px 5px 5px 0px;
    }
    .error-message {
        background-color: #FADBD8;
        border-left: 5px solid #E74C3C;
        padding: 10px;
        border-radius: 0px 5px 5px 0px;
    }
    .info-card {
        background-color: #EBF5FB;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .metric-card {
        background-color: #F8F9F9;
        border-left: 4px solid #3498DB;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
    }
    .sidebar .sidebar-content {
        background-color: #F8F9F9;
    }
</style>
""", unsafe_allow_html=True)

# Titre de l'application et auteur
st.markdown('<h1 class="main-header">Mining Geology Data Application</h1>', unsafe_allow_html=True)
st.markdown('<p class="author">Développé par: Didier Ouedraogo, P.Geo.</p>', unsafe_allow_html=True)

# Fonction pour convertir les chaînes en nombres flottants avec gestion d'erreurs
def safe_float(value):
    if pd.isna(value):
        return 0.0
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0

# Fonction pour télécharger les données en CSV
def get_csv_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" style="color: #3498DB; text-decoration: none;"><button style="background-color: #3498DB; color: white; padding: 8px 15px; border: none; border-radius: 5px; cursor: pointer;">Télécharger {text}</button></a>'
    return href

# Fonction pour vérifier si une colonne existe dans un DataFrame
def column_exists(df, column_name):
    return df is not None and column_name and column_name in df.columns

# Fonction pour créer des composites d'analyses avec coordonnées
def create_composites(assays_df, hole_id_col, from_col, to_col, value_col, composite_length=1.0, 
                     collars_df=None, survey_df=None, x_col=None, y_col=None, z_col=None, 
                     azimuth_col=None, dip_col=None, depth_col=None):
    if assays_df is None or assays_df.empty:
        return None
    
    # Vérifier que toutes les colonnes nécessaires existent
    if not all(col in assays_df.columns for col in [hole_id_col, from_col, to_col, value_col]):
        st.markdown('<div class="error-message">Colonnes manquantes dans le DataFrame des analyses</div>', unsafe_allow_html=True)
        return None
    
    # Créer une copie des données pour éviter de modifier l'original
    df = assays_df.copy()
    
    # Convertir les colonnes numériques en flottants
    for col in [from_col, to_col, value_col]:
        df[col] = df[col].apply(safe_float)
    
    # Initialiser le DataFrame des composites
    composites = []
    
    # Pour chaque trou de forage
    for hole_id in df[hole_id_col].unique():
        hole_data = df[df[hole_id_col] == hole_id].sort_values(by=from_col)
        
        if hole_data.empty:
            continue

        # Récupérer les données de collars et survey pour les coordonnées si disponibles
        collar_info = None
        surveys = None
        
        if collars_df is not None and survey_df is not None and all(col is not None for col in [x_col, y_col, z_col, depth_col, azimuth_col, dip_col]):
            if hole_id in collars_df[hole_id_col].values:
                collar_info = collars_df[collars_df[hole_id_col] == hole_id].iloc[0]
                surveys = survey_df[survey_df[hole_id_col] == hole_id].sort_values(by=depth_col)
        
        # Pour chaque intervalle de composite
        composite_start = float(hole_data[from_col].min())
        while composite_start < float(hole_data[to_col].max()):
            composite_end = composite_start + composite_length
            
            # Trouver tous les intervalles qui chevauchent le composite actuel
            overlapping = hole_data[
                ((hole_data[from_col] >= composite_start) & (hole_data[from_col] < composite_end)) |
                ((hole_data[to_col] > composite_start) & (hole_data[to_col] <= composite_end)) |
                ((hole_data[from_col] <= composite_start) & (hole_data[to_col] >= composite_end))
            ]
            
            if not overlapping.empty:
                # Calculer le poids pondéré pour chaque intervalle chevauchant
                weighted_values = []
                total_length = 0
                
                for _, row in overlapping.iterrows():
                    overlap_start = max(composite_start, row[from_col])
                    overlap_end = min(composite_end, row[to_col])
                    overlap_length = overlap_end - overlap_start
                    
                    if overlap_length > 0:
                        weighted_values.append(row[value_col] * overlap_length)
                        total_length += overlap_length
                
                # Calculer la valeur pondérée du composite
                if total_length > 0:
                    composite_value = sum(weighted_values) / total_length
                    
                    # Créer une entrée de composite de base
                    composite_entry = {
                        hole_id_col: hole_id,
                        'From': composite_start,
                        'To': composite_end,
                        'Length': total_length,
                        value_col: composite_value
                    }
                    
                    # Ajouter les coordonnées si les données nécessaires sont disponibles
                    if collar_info is not None and not surveys.empty:
                        # Calculer la position moyenne (milieu de l'intervalle)
                        mid_depth = (composite_start + composite_end) / 2
                        
                        # Chercher les données de survey les plus proches
                        closest_idx = surveys[depth_col].apply(lambda d: abs(d - mid_depth)).idxmin()
                        closest_survey = surveys.loc[closest_idx]
                        
                        # Récupérer les données du collar
                        x_start = safe_float(collar_info[x_col])
                        y_start = safe_float(collar_info[y_col])
                        z_start = safe_float(collar_info[z_col])
                        
                        # Calculer les coordonnées 3D approximatives pour le composite
                        # (Méthode simplifiée - pour une précision parfaite, une interpolation plus complexe serait nécessaire)
                        depth = safe_float(closest_survey[depth_col])
                        azimuth = safe_float(closest_survey[azimuth_col])
                        dip = safe_float(closest_survey[dip_col])
                        
                        # Convertir l'azimuth et le dip en direction 3D
                        azimuth_rad = np.radians(azimuth)
                        dip_rad = np.radians(dip)
                        
                        # Calculer la position approximative
                        dx = depth * np.sin(dip_rad) * np.sin(azimuth_rad)
                        dy = depth * np.sin(dip_rad) * np.cos(azimuth_rad)
                        dz = -depth * np.cos(dip_rad)  # Z est négatif pour la profondeur
                        
                        # Ajouter les coordonnées au composite
                        composite_entry['X'] = x_start + dx
                        composite_entry['Y'] = y_start + dy
                        composite_entry['Z'] = z_start + dz
                    
                    # Ajouter le composite au résultat
                    composites.append(composite_entry)
            
            composite_start = composite_end
    
    # Créer un DataFrame à partir des composites
    if composites:
        return pd.DataFrame(composites)
    else:
        return pd.DataFrame()

# Fonction pour créer un strip log pour un forage spécifique
def create_strip_log(hole_id, collars_df, survey_df, lithology_df, assays_df, 
                    hole_id_col, depth_col, 
                    lith_from_col, lith_to_col, lith_col,
                    assay_from_col, assay_to_col, assay_value_col):
    
    # Vérifier si les données nécessaires sont disponibles
    if collars_df is None or survey_df is None:
        return None
    
    # Récupérer les informations du forage
    hole_surveys = survey_df[survey_df[hole_id_col] == hole_id].sort_values(by=depth_col)
    
    if hole_surveys.empty:
        return None
    
    # Convertir les valeurs de profondeur en flottants
    hole_surveys[depth_col] = hole_surveys[depth_col].apply(safe_float)
    
    # Profondeur maximale du forage
    max_depth = hole_surveys[depth_col].max()
    
    # Créer la figure
    fig, axes = plt.subplots(1, 3, figsize=(12, max_depth/10 + 2), 
                            gridspec_kw={'width_ratios': [2, 1, 3]})
    
    # Titre du graphique
    fig.suptitle(f'Strip Log - Forage {hole_id}', fontsize=16)
    
    # 1. Colonne de lithologie
    if lithology_df is not None and all(col and col in lithology_df.columns for col in [hole_id_col, lith_from_col, lith_to_col, lith_col]):
        hole_litho = lithology_df[lithology_df[hole_id_col] == hole_id].sort_values(by=lith_from_col)
        
        if not hole_litho.empty:
            # Convertir les colonnes de profondeur en flottants
            hole_litho[lith_from_col] = hole_litho[lith_from_col].apply(safe_float)
            hole_litho[lith_to_col] = hole_litho[lith_to_col].apply(safe_float)
            
            # Définir une palette de couleurs pour les différentes lithologies
            unique_liths = hole_litho[lith_col].unique()
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_liths)))
            lith_color_map = {lith: color for lith, color in zip(unique_liths, colors)}
            
            # Dessiner des rectangles pour chaque intervalle de lithologie
            for _, row in hole_litho.iterrows():
                lith_from = row[lith_from_col]
                lith_to = row[lith_to_col]
                lith_type = row[lith_col]
                
                axes[0].add_patch(plt.Rectangle((0, lith_from), 1, lith_to - lith_from, 
                                                color=lith_color_map[lith_type]))
                
                # Ajouter le texte de la lithologie au milieu de l'intervalle
                interval_height = lith_to - lith_from
                font_size = min(10, max(6, interval_height * 0.8))  # Taille de police adaptative
                
                axes[0].text(0.5, (lith_from + lith_to) / 2, lith_type,
                            ha='center', va='center', fontsize=font_size,
                            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
    
    # Configurer l'axe de la lithologie
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(max_depth, 0)  # Inverser l'axe y pour que la profondeur augmente vers le bas
    axes[0].set_xlabel('Lithologie')
    axes[0].set_ylabel('Profondeur (m)')
    axes[0].set_xticks([])
    
    # 2. Colonne de profondeur
    depth_ticks = np.arange(0, max_depth + 10, 10)
    axes[1].set_yticks(depth_ticks)
    axes[1].set_ylim(max_depth, 0)
    axes[1].set_xlim(0, 1)
    axes[1].set_xticks([])
    axes[1].set_xlabel('Profondeur')
    axes[1].grid(axis='y')
    
    # 3. Colonne d'analyses
    if assays_df is not None and all(col and col in assays_df.columns for col in [hole_id_col, assay_from_col, assay_to_col, assay_value_col]):
        hole_assays = assays_df[assays_df[hole_id_col] == hole_id].sort_values(by=assay_from_col)
        
        if not hole_assays.empty:
            # Convertir les colonnes numériques en flottants
            hole_assays[assay_from_col] = hole_assays[assay_from_col].apply(safe_float)
            hole_assays[assay_to_col] = hole_assays[assay_to_col].apply(safe_float)
            hole_assays[assay_value_col] = hole_assays[assay_value_col].apply(safe_float)
            
            # Trouver la valeur maximale pour normaliser
            max_value = hole_assays[assay_value_col].max()
            
            # Dessiner des barres horizontales pour chaque intervalle d'analyse
            for _, row in hole_assays.iterrows():
                assay_from = row[assay_from_col]
                assay_to = row[assay_to_col]