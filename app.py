import streamlit as st
import pandas as pd
import numpy as np
import faiss
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag


# Automatické stažení potřebných NLTK balíčků při prvním spuštění
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('punkt')
    # Pro jistotu stáhneme obě verze, starou i novou, ať je klid
    nltk.download('averaged_perceptron_tagger')
    nltk.download('averaged_perceptron_tagger_eng')

# Nastavení stránky
st.set_page_config(page_title="Filmový archiv", layout="wide")

# CSS: Vintage minimalismus zachován
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,600;1,400&family=Space+Mono&display=swap');

    .stApp {
        background-color: #24221f;
        background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)' opacity='0.02'/%3E%3C/svg%3E");
        color: #e6e2d8;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    h1, h2, h3 { font-family: 'Lora', serif; font-weight: 400; letter-spacing: -0.02em; color: #f2efe9; }
    h1 { border-bottom: 1px solid #3d3a35; padding-bottom: 0.5rem; margin-bottom: 2rem; }
    
    .stTextInput > div > div > input, .stSelectbox > div > div > div {
        background-color: #1a1816 !important; color: #e6e2d8 !important;
        border: 1px solid #4a463e !important; border-radius: 2px !important; 
        font-family: 'Space Mono', monospace; font-size: 0.9rem;
    }
    div[role="radiogroup"] label p { color: #e6e2d8 !important; font-family: 'Space Mono', monospace; font-size: 0.95rem; }
    .stTextInput > div > div > input:focus, .stSelectbox > div > div > div:focus { border-color: #c96245 !important; box-shadow: none !important; }
    
    div.stButton > button:first-child {
        background-color: #24221f; color: #c96245; border: 1px solid #c96245; border-radius: 0;
        font-family: 'Space Mono', monospace; text-transform: uppercase; letter-spacing: 0.05em; transition: none;
    }
    div.stButton > button:first-child:hover { background-color: #c96245; color: #1a1816; border: 1px solid #c96245; }
    
    /* Zvláštní styl pro sekundární tlačítka (klastry a stránkování) */
    .stButton [data-testid="baseButton-secondary"] {
        border-color: #4a463e; color: #cfc8b6; font-size: 0.8rem; padding: 0.2rem 0.5rem;
    }
    .stButton [data-testid="baseButton-secondary"]:hover { border-color: #e09f3e; color: #e09f3e; background-color: transparent;}

    .streamlit-expanderHeader { font-family: 'Space Mono', monospace; font-size: 0.85rem; color: #cfc8b6; border-bottom: 1px dashed #4a463e; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.title("Archiv sémantických rezonancí")

# --- INITIALIZACE DAT A MODELŮ ---
@st.cache_resource
def load_data_and_models():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    df = pd.read_pickle("movie_database.pkl")
    index = faiss.read_index("faiss_index.bin")
    
    # Detekce sloupce s klastry (fallback pokud v PKL náhodou chybí)
    cluster_col = 'Cluster_384D' if 'Cluster_384D' in df.columns else ('Cluster' if 'Cluster' in df.columns else None)
    
    # Trénink TF-IDF Vectorizeru pro Jazykovou lupu
    custom_stop_words = list(ENGLISH_STOP_WORDS) + ['film', 'movie', 'story', 'tells', 'life', 'man', 'woman', 'young', 'new', 'time', 'day', 'boy', 'girl', 'father', 'mother', 'family', 'named', 'role', 'plays']
    vectorizer = TfidfVectorizer(stop_words=custom_stop_words, max_df=0.6, max_features=10000)
    vectorizer.fit(df['Plot'])
    
    return model, df, index, vectorizer, cluster_col

model, df, index, vectorizer, cluster_col = load_data_and_models()

def filter_proper_nouns(text):
    # Rozsekání textu na slova a přiřazení slovních druhů
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    
    # Ponecháme pouze slova, která NEJSOU vlastní jména (NNP, NNPS)
    # Také ignorujeme krátká slova a interpunkci
    filtered_words = [word for word, tag in tagged if tag not in ('NNP', 'NNPS') and len(word) > 2]
    return " ".join(filtered_words)

def get_keywords(plot_text, vec, top_n=5):
    # 1. Odstranění jmen postav (Ravi, John, Matrix...)
    clean_text = filter_proper_nouns(plot_text)
    
    # 2. Výpočet TF-IDF pouze nad sémantickými slovy
    tfidf_scores = vec.transform([clean_text]).toarray()[0]
    top_indices = tfidf_scores.argsort()[-top_n:][::-1]
    feature_names = vec.get_feature_names_out()
    
    return [feature_names[i].upper() for i in top_indices if tfidf_scores[i] > 0]

# --- SPRÁVA STAVU APLIKACE ---
if 'search_performed' not in st.session_state:
    st.session_state.search_performed = False
    st.session_state.all_results = []
    st.session_state.page = 1
    st.session_state.active_cluster = None

col_input, col_viz = st.columns([1, 1.2], gap="large")

with col_input:
    st.markdown("### Vstupní parametry")
    
    search_mode = st.radio("Metoda hledání:", ["Sémantický popis děje", "Asociace k existujícímu dílu"], label_visibility="collapsed")
    
    if search_mode == "Sémantický popis děje":
        query = st.text_input("Popiš zápletku nebo motiv (EN):", value="A lone detective investigating a murder in a dystopian city")
    else:
        all_titles = sorted(df['Title'].unique())
        selected_movie = st.selectbox("Vyberte referenční snímek (EN):", all_titles)
        query = df[df['Title'] == selected_movie].iloc[0]['Plot']
        st.markdown("<span style='color:#cfc8b6; font-size:0.85rem;'>Referenční text načten.</span>", unsafe_allow_html=True)

    # NOVÉ: Sémantická negace
    neg_query = st.text_input("Sémantická negace (Co ve filmu NECHCEŠ):", value="")

    # NOVÉ: Časová osa
    min_year, max_year = int(df['Release Year'].min()), int(df['Release Year'].max())
    selected_years = st.slider("Časová osa (Rok vydání):", min_value=min_year, max_value=max_year, value=(min_year, max_year))

    english_boost = st.slider("Zvýhodnění anglofonní produkce", min_value=0.0, max_value=0.2, value=0.05, step=0.01)
    
    if st.button("Iniciovat hledání"):
        # Reset stavů pro nové hledání
        st.session_state.page = 1
        st.session_state.active_cluster = None
        st.session_state.search_performed = True
        
        # 1. Kódování hlavního dotazu
        query_vector = model.encode([query]).astype('float32')
        
        # 2. Vektorová matematika: Odečtení negace
        if neg_query.strip():
            neg_vector = model.encode([neg_query]).astype('float32')
            query_vector = query_vector - neg_vector
            
        # Nutná re-normalizace po odečtení vektorů pro sférický FAISS
        faiss.normalize_L2(query_vector)
        
        # Vytáhneme víc výsledků (např. 150), abychom měli rezervu po aplikaci časového filtru
        distances, indices = index.search(query_vector, 150)
        
        raw_results = []
        seen_titles = set()
        
        for i in range(150):
            idx = indices[0][i]
            title = df.iloc[idx]['Title']
            year = df.iloc[idx]['Release Year']
            
            # Filtrování duplikátů, referenčního filmu a časové osy
            if title in seen_titles or (search_mode != "Sémantický popis děje" and title == selected_movie): continue
            if not (selected_years[0] <= year <= selected_years[1]): continue
                
            seen_titles.add(title)
            sim_score = distances[0][i]
            is_eng = df.iloc[idx]['is_english']
            final_score = sim_score + (english_boost if is_eng else 0)
            movie_cluster = df.iloc[idx][cluster_col] if cluster_col else None
            
            raw_results.append({
                'idx': idx, 'Title': title, 'Year': year, 'Score': final_score, 
                'Plot': df.iloc[idx]['Plot'], 'Wiki': df.iloc[idx]['Wiki Page'], 
                'Is_Eng': is_eng, 'Cluster': movie_cluster
            })
            
        # Uložení všech vyfiltrovaných výsledků do paměti relace
        st.session_state.all_results = sorted(raw_results, key=lambda x: x['Score'], reverse=True)

    # --- VYKRESLENÍ VÝSLEDKŮ A STRÁNKOVÁNÍ ---
    if st.session_state.search_performed and st.session_state.all_results:
        st.markdown("<br>### Nalezené shody", unsafe_allow_html=True)
        
        current_limit = st.session_state.page * 5
        display_results = st.session_state.all_results[:current_limit]
        
        for r in display_results:
            tag = "ENG" if r['Is_Eng'] else "INTL"
            
            # NOVÉ: Jazyková lupa (Extrakce TF-IDF klíčových slov)
            keywords = get_keywords(r['Plot'], vectorizer)
            kw_string = " &nbsp;&middot;&nbsp; ".join([f"<span style='color:#cfc8b6;'>{k}</span>" for k in keywords])
            
            st.markdown(f"""
            <div style="border-left: 3px solid #c96245; padding-left: 1rem; margin-bottom: 0.5rem;">
                <h4 style="margin: 0; color: #e6e2d8; font-family: -apple-system, sans-serif;">{r['Title']} <span style="color:#9e9789; font-weight:normal;">({r['Year']})</span></h4>
                <div style="font-family: 'Space Mono', monospace; font-size: 0.8rem; color: #c96245; margin-top: 0.2rem; margin-bottom: 0.2rem;">
                    [ {tag} ] SHODA: {r['Score']:.3f} &nbsp;&mdash;&nbsp; <a href="{r['Wiki']}" style="color: #c96245; text-decoration: none;">WIKI LOKACE</a>
                </div>
                <div style="font-family: 'Space Mono', monospace; font-size: 0.75rem; letter-spacing: 0.05em; margin-bottom: 0.5rem;">
                    {kw_string}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            col_btn, col_exp = st.columns([1, 2])
            with col_btn:
                # NOVÉ: Skok do klastru
                if r['Cluster'] is not None:
                    if st.button("Zobrazit topologii žánru", key=f"btn_cluster_{r['idx']}", type="secondary"):
                        st.session_state.active_cluster = r['Cluster']
                        st.rerun()
            with col_exp:
                with st.expander("Číst záznam zápletky"):
                    st.write(r['Plot'])
            st.markdown("<div style='margin-bottom: 1.5rem;'></div>", unsafe_allow_html=True)

        # NOVÉ: Tlačítko pro načtení dalších výsledků
        if current_limit < len(st.session_state.all_results):
            if st.button("Načíst dalších 5 korelací", type="secondary"):
                st.session_state.page += 1
                st.rerun()

with col_viz:
    st.markdown("### Mapa archivu")
    
    # Optimalizace: pro základní mapu vykreslíme menší vzorek, aby web nelagoval
    df_sample = df.sample(1500, random_state=42).copy()
    
    fig = go.Figure()
    
    # 1. Podkladová vrstva (Šedý prach)
    fig.add_trace(go.Scatter(
        x=df_sample['x'], y=df_sample['y'], mode='markers',
        marker=dict(size=4, color='#5c564b', opacity=0.3, line=dict(width=0)),
        text=df_sample['Title'], hoverinfo='text', name='Archiv'
    ))
    
    # 2. Vrstva aktivního klastru (Zlatavý ostrov)
    if st.session_state.active_cluster is not None and cluster_col is not None:
        cluster_data = df[df[cluster_col] == st.session_state.active_cluster]
        fig.add_trace(go.Scatter(
            x=cluster_data['x'], y=cluster_data['y'], mode='markers',
            marker=dict(size=6, color='#e09f3e', opacity=0.8, line=dict(width=0)),
            text=cluster_data['Title'], hoverinfo='text', name='Aktivní žánr'
        ))
    
    # 3. Vrstva hledaných shod (Červené čtverce)
    if st.session_state.search_performed and st.session_state.all_results:
        # Zvýrazníme jen ty, které uživatel aktuálně vidí na obrazovce
        current_limit = st.session_state.page * 5
        visible_indices = [r['idx'] for r in st.session_state.all_results[:current_limit]]
        df_highlight = df.iloc[visible_indices]
        
        fig.add_trace(go.Scatter(
            x=df_highlight['x'], y=df_highlight['y'], mode='markers+text',
            marker=dict(size=10, color='#c96245', symbol='square', line=dict(color='#24221f', width=2)),
            text=df_highlight['Title'], hoverinfo='text', textposition="top center",
            textfont=dict(family="Space Mono", color="#e6e2d8", size=11), name='Shoda'
        ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, b=0, t=0),
        xaxis=dict(visible=True, showgrid=True, gridcolor='rgba(255,255,255,0.05)', zeroline=False, showticklabels=False, title=''),
        yaxis=dict(visible=True, showgrid=True, gridcolor='rgba(255,255,255,0.05)', zeroline=False, showticklabels=False, title=''),
        showlegend=False, dragmode='pan', hovermode='closest', height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)