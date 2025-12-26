import streamlit as st
import pandas as pd
import requests
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import heapq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Movie Recommender", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stButton>button {
        width: 100%; border-radius: 5px; height: 3em;
        background-color: #B20710; color: white; font-weight: bold; border: none;
    }
    .stButton>button:hover { background-color: #E50914; color: white; }
    h1 { color: #B20710; text-align: center; font-family: 'Arial Black', sans-serif; }
    
    .movie-card { 
        border-radius: 10px; 
        background-color: #1a1c23; 
        padding: 10px; 
        text-align: center; 
        height: 100%;
        transition: transform 0.3s ease, border 0.3s ease;
        border: 1px solid transparent;
    }
    
    .movie-card:hover { 
        transform: scale(1.08);
        border: 1px solid #B20710;
        z-index: 10;
        box-shadow: 0px 10px 20px rgba(178, 7, 16, 0.4);
    }
    
    label { color: white !important; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_data():
    df = pd.read_csv('movies_cleaned.csv')
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['tags'])
    sim = cosine_similarity(tfidf_matrix)
    return df, sim

df, similarity = load_data()

def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
    try:
        data = requests.get(url).json()
        return "https://image.tmdb.org/t/p/w500/" + data['poster_path']
    except:
        return "https://via.placeholder.com/500x750?text=No+Poster"

def get_neighbors(movie_idx, top_k=10):
    sim_scores = list(enumerate(similarity[movie_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    return sim_scores[1:top_k+1]

def astar_search(start_idx, target_idx):
    queue = [(0 + (1 - similarity[start_idx][target_idx]), 0, start_idx, [start_idx])]
    visited = {}
    while queue:
        (f, g, vertex, path) = heapq.heappop(queue)
        if vertex == target_idx: return path, int(round(g * 100))
        if vertex in visited and visited[vertex] <= g: continue
        visited[vertex] = g
        for neighbor, sim in get_neighbors(vertex):
            new_g = g + (1 - sim)
            h = 1 - similarity[neighbor][target_idx]
            heapq.heappush(queue, (new_g + h, new_g, neighbor, path + [neighbor]))
    return None, 0

def ucs_search(start_idx, target_idx):
    queue = [(0, start_idx, [start_idx])]
    visited = {}
    while queue:
        (cost, vertex, path) = heapq.heappop(queue)
        if vertex == target_idx: return path, int(round(cost * 100))
        if vertex in visited and visited[vertex] <= cost: continue
        visited[vertex] = cost
        for neighbor, sim in get_neighbors(vertex):
            new_cost = cost + (1 - sim)
            heapq.heappush(queue, (new_cost, neighbor, path + [neighbor]))
    return None, 0

def dfs_search(start_idx, target_idx, max_depth=30):
    stack = [(start_idx, [start_idx])]
    visited = set()
    while stack:
        (vertex, path) = stack.pop()
        if vertex == target_idx: return path
        if vertex not in visited and len(path) < max_depth:
            visited.add(vertex)
            for neighbor, _ in get_neighbors(vertex, top_k=10):
                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor]))
    return None

st.markdown("<h1>AI Movie Recommender</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.header("Visualization")
    show_analysis = st.checkbox("Show AI Analysis (Sidebar)")
    target_movie = st.selectbox("Search Target Movie:", df['title'].values, index=5)
    algo_type = st.radio("Algorithm:", ["A*", "UCS", "DFS"])

selected_movie = st.selectbox("Select a movie you like:", df['title'].values)

if st.button('Show Recommendations'):
    idx = df[df['title'] == selected_movie].index[0]
    
    distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])[1:7]
    st.write(f"### Results for: {selected_movie}")
    cols = st.columns(6)
    for i, m in enumerate(distances):
        with cols[i]:
            movie_id = df.iloc[m[0]].movie_id
            st.markdown(f'''
                <div class="movie-card">
                    <img src="{fetch_poster(movie_id)}" style="width:100%; border-radius:5px;">
                    <p style="color:white; font-size:0.8rem; margin-top:10px; font-weight:bold;">{df.iloc[m[0]].title}</p>
                </div>
            ''', unsafe_allow_html=True)

    if show_analysis:
        target_idx = df[df['title'] == target_movie].index[0]
        path, cost = None, 0
        
        if algo_type == "A*": path, cost = astar_search(idx, target_idx)
        elif algo_type == "UCS": path, cost = ucs_search(idx, target_idx)
        else: 
            path = dfs_search(idx, target_idx)
            if path:
                raw_cost = sum([1 - similarity[path[i]][path[i+1]] for i in range(len(path)-1)])
                cost = int(round(raw_cost * 100))
        
        if path:
            st.sidebar.success(f"Algorithm: {algo_type}")
            st.sidebar.info(f"Path Length: {len(path)} steps")
            st.sidebar.warning(f"Total Cost: {cost}")
            
            path_sims = [similarity[path[i]][path[i+1]] for i in range(len(path)-1)]
            titles = df.iloc[path]['title'].tolist()

            st.sidebar.write(" Similarity Heatmap")
            fig1, ax1 = plt.subplots(figsize=(max(8, len(path)*0.5), max(6, len(path)*0.4)))
            sns.heatmap(similarity[np.ix_(path, path)], annot=(len(path)<=20), xticklabels=titles, yticklabels=titles, cmap='Reds', ax=ax1)
            ax1.set_xlabel("Steps (X-Axis)")
            ax1.set_ylabel("Steps (Y-Axis)")
            plt.xticks(rotation=45, ha='right')
            st.sidebar.pyplot(fig1)

            st.sidebar.write(" Similarity Bar Chart")
            fig_bar, ax_bar = plt.subplots()
            ax_bar.bar([f"S{i+1}" for i in range(len(path_sims))], path_sims, color='#B20710')
            ax_bar.set_xlabel("Step Number (X-Axis)")
            ax_bar.set_ylabel("Similarity Score (Y-Axis)")
            st.sidebar.pyplot(fig_bar)

            st.sidebar.write(" Path Similarity Flow")
            fig_line, ax_line = plt.subplots()
            ax_line.plot(path_sims, marker='o', color='#B20710', linewidth=2, markersize=6)
            ax_line.set_xlabel("Transition Step (X-Axis)")
            ax_line.set_ylabel("Cosine Similarity (Y-Axis)")
            ax_line.grid(True, alpha=0.3)
            st.sidebar.pyplot(fig_line)
            
        else:
            st.sidebar.error("No path found.")