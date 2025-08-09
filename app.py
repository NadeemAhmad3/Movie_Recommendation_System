# --- Import Core Libraries ---
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# --- Page Configuration ---
st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Custom CSS ---
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")


# --- DATA & MODEL LOADING (Cached for Performance) ---
@st.cache_data(show_spinner="Loading models and data...")
def load_artifacts():
    """
    Load all necessary saved models and dataframes from the 'saved_models' directory.
    This function is cached to prevent reloading on every interaction.
    """
    artifacts = {}
    model_dir = 'saved_models'
    
    # List of expected files to load
    files_to_load = [
        'movies.pkl', 'train_matrix.pkl', 'user_means.pkl', 
        'user_similarity_df.pkl', 'item_similarity_df.pkl', 'svd_model.pkl', 
        'svd_preds_df.pkl', 'results_df.pkl', 'demographics_fig.pkl', 
        'numerical_fig.pkl', 'correlation_fig.pkl', 'behavioral_fig.pkl', 
        'svd_variance_fig.pkl'
    ]
    
    for filename in files_to_load:
        path = os.path.join(model_dir, filename)
        key = filename.split('.')[0]
        try:
            with open(path, 'rb') as f:
                artifacts[key] = pickle.load(f)
        except FileNotFoundError:
            st.error(f"Error: The file '{filename}' was not found in the '{model_dir}' directory.")
            st.stop()
            
    # Also load the original full dataset for context
    try:
        ratings = pd.read_csv('input/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
        users = pd.read_csv("input/u.user", sep="|", names=["user_id", "age", "gender", "occupation", "zip_code"])
        data = pd.merge(pd.merge(ratings, artifacts['movies']), users)
        artifacts['full_data'] = data
    except FileNotFoundError:
        st.error("Error: Could not find 'u.data' or 'u.user' in the 'input' directory.")
        st.stop()
        
    return artifacts

artifacts = load_artifacts()

# --- Unpack artifacts for easier access ---
movies_df = artifacts.get('movies')
train_matrix = artifacts.get('train_matrix')
user_means = artifacts.get('user_means')
user_similarity_df = artifacts.get('user_similarity_df')
item_similarity_df = artifacts.get('item_similarity_df')
svd_model = artifacts.get('svd_model')
svd_preds_df = artifacts.get('svd_preds_df')
results_df = artifacts.get('results_df')
full_data = artifacts.get('full_data')


# --- RECOMMENDATION LOGIC ---

def predict_user_cf_rating(user_id, movie_id, k=50):
    if user_id not in user_similarity_df.index or movie_id not in train_matrix.columns:
        return user_means.get(user_id, 3.0)
    similar_users = user_similarity_df[user_id].sort_values(ascending=False).index[1:]
    movie_ratings = train_matrix[movie_id]
    top_similar_raters = similar_users.intersection(movie_ratings[movie_ratings > 0].index)[:k]
    if top_similar_raters.empty:
        return user_means.get(user_id, 3.0)
    top_similarities = user_similarity_df.loc[user_id, top_similar_raters]
    top_ratings = train_matrix.loc[top_similar_raters, movie_id]
    numerator = np.dot(top_ratings, top_similarities)
    denominator = top_similarities.sum()
    return np.clip(numerator / denominator, 1, 5) if denominator != 0 else user_means.get(user_id, 3.0)

def get_user_cf_recommendations(user_id, n=10, k_similar_users=100):
    if user_id not in train_matrix.index: return pd.DataFrame()
    top_k_similar_users = user_similarity_df[user_id].sort_values(ascending=False).index[1:k_similar_users+1]
    candidate_movies_ratings = train_matrix.loc[top_k_similar_users]
    candidate_movies_ratings = candidate_movies_ratings.loc[:, (candidate_movies_ratings > 0).sum() > 2]
    user_rated_movies = train_matrix.loc[user_id][train_matrix.loc[user_id] > 0].index
    candidate_movies = [movie for movie in candidate_movies_ratings.columns if movie not in user_rated_movies]
    predictions = [(movie_id, predict_user_cf_rating(user_id, movie_id)) for movie_id in candidate_movies]
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_n_recs = predictions[:n]
    if not top_n_recs: return pd.DataFrame()
    rec_ids, rec_scores = zip(*top_n_recs)
    rec_df = pd.DataFrame({'movie_id': rec_ids, 'predicted_rating': rec_scores})
    return rec_df.merge(movies_df, on='movie_id', how='left')

def predict_item_cf_rating(user_id, movie_id, k=50):
    if user_id not in train_matrix.index or movie_id not in item_similarity_df.index: return 3.0
    user_ratings = train_matrix.loc[user_id]
    rated_items_by_user = user_ratings[user_ratings > 0].index
    similarities_to_rated_items = item_similarity_df.loc[movie_id, rated_items_by_user]
    top_k_similar_items = similarities_to_rated_items.nlargest(k)
    if top_k_similar_items.empty: return 3.0
    ratings_for_top_items = user_ratings.loc[top_k_similar_items.index]
    numerator = np.dot(ratings_for_top_items, top_k_similar_items)
    denominator = top_k_similar_items.sum()
    return np.clip(numerator / denominator, 1, 5) if denominator != 0 else 3.0

def get_item_cf_recommendations(user_id, n=10):
    if user_id not in train_matrix.index: return pd.DataFrame()
    user_ratings = train_matrix.loc[user_id]
    highly_rated_items = user_ratings[user_ratings >= 4].index
    candidate_items = set()
    for item in highly_rated_items:
        similar_items = item_similarity_df[item].nlargest(15).index
        candidate_items.update(similar_items)
    rated_items = user_ratings[user_ratings > 0].index
    candidate_items.difference_update(rated_items)
    if not candidate_items: return pd.DataFrame()
    predictions = [(movie_id, predict_item_cf_rating(user_id, movie_id)) for movie_id in candidate_items]
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_n_recs = predictions[:n]
    if not top_n_recs: return pd.DataFrame()
    rec_ids, rec_scores = zip(*top_n_recs)
    rec_df = pd.DataFrame({'movie_id': rec_ids, 'predicted_rating': rec_scores})
    return rec_df.merge(movies_df, on='movie_id', how='left')

def get_svd_recommendations(user_id, n=10):
    if user_id not in svd_preds_df.index: return pd.DataFrame()
    user_predictions = svd_preds_df.loc[user_id]
    user_rated_movies = train_matrix.loc[user_id][train_matrix.loc[user_id] > 0].index
    unseen_predictions = user_predictions.drop(user_rated_movies).sort_values(ascending=False)
    top_n_recs = unseen_predictions.head(n).reset_index()
    top_n_recs.columns = ['movie_id', 'predicted_rating']
    return top_n_recs.merge(movies_df, on='movie_id', how='left')

def get_hybrid_recommendations(user_id, n=10, weights={'user': 0.3, 'item': 0.3, 'svd': 0.4}):
    user_recs = get_user_cf_recommendations(user_id, n=20)
    item_recs = get_item_cf_recommendations(user_id, n=20)
    svd_recs = get_svd_recommendations(user_id, n=20)
    combined_scores = {}
    
    def add_to_scores(recs_df, weight):
        if recs_df.empty: return
        min_score, max_score = recs_df['predicted_rating'].min(), recs_df['predicted_rating'].max()
        recs_df['norm_score'] = (recs_df['predicted_rating'] - min_score) / (max_score - min_score) if max_score > min_score else 1.0
        for _, row in recs_df.iterrows():
            combined_scores[row['movie_id']] = combined_scores.get(row['movie_id'], 0) + (row['norm_score'] * weight)
            
    add_to_scores(user_recs, weights['user'])
    add_to_scores(item_recs, weights['item'])
    add_to_scores(svd_recs, weights['svd'])
    
    if not combined_scores: return pd.DataFrame()
    sorted_recs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:n]
    rec_ids, final_scores = zip(*sorted_recs)
    rec_df = pd.DataFrame({'movie_id': rec_ids, 'combined_score': final_scores})
    return rec_df.merge(movies_df, on='movie_id', how='left')


# --- UI & PAGE RENDERING ---

# --- UI & PAGE RENDERING ---

# REPLACE the old show_welcome_page function with this new one
def show_welcome_page():
    # 1. Add a high-quality, thematic image banner
 # This is the corrected line
    st.image(
    "https://images.pexels.com/photos/7991579/pexels-photo-7991579.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2",
    use_container_width=True
    )

    # 2. A more engaging title and subtitle
    st.title("Welcome to FLIX-AI")
    st.markdown("### Your Personal Guide to the World of Cinema")
    st.write(
        "This application demonstrates a powerful movie recommendation system built with real-world data. "
        "Explore the sidebar to see data insights, model performance, or get your next movie recommendation!"
    )
    st.divider()

    # 3. Create interactive feature cards in columns
    st.header("Explore the Project")
    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.markdown("""
        <div class="feature-card">
            <span class="feature-icon"><i class="bi bi-graph-up-arrow"></i></span>
            <h3 class="feature-title">Data Analysis</h3>
            <p class="feature-description">
                Dive deep into the MovieLens dataset. Visualize user demographics, rating distributions, and behavioral patterns.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <span class="feature-icon"><i class="bi bi-stars"></i></span>
            <h3 class="feature-title">AI Recommendations</h3>
            <p class="feature-description">
                Get personalized movie suggestions from multiple  Collaborative Filtering and SVD.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card">
            <span class="feature-icon"><i class="bi bi-film"></i></span>
            <h3 class="feature-title">Find Similar Movies</h3>
            <p class="feature-description">
                Loved a movie? Select it and let our system instantly find other movies with similar rating patterns from users.
            </p>
        </div>
        """, unsafe_allow_html=True)

def show_eda_page():
    st.title("📊 Exploratory Data Analysis (EDA)")
    st.write("This section visualizes the key insights derived from the MovieLens dataset.")

    st.subheader("User Demographics")
    st.pyplot(artifacts['demographics_fig'])
    st.markdown("""
    **Observations:**
    - The dataset is predominantly male (over two-thirds).
    - The most common occupations are 'student', 'other', and 'educator'. This provides context on the user base's likely interests and available time for watching movies.
    """)

    st.subheader("Numerical Variable Analysis")
    st.pyplot(artifacts['numerical_fig'])
    st.markdown("""
    **Observations:**
    - **Age:** The user age distribution is skewed towards younger individuals, with a peak in the 20-30 age range.
    - **Rating:** Ratings are left-skewed, with '4' being the most frequent rating. This indicates users are generally more likely to rate movies they enjoyed.
    """)
    
    st.subheader("Rating Behavior Analysis")
    st.pyplot(artifacts['behavioral_fig'])
    st.markdown("""
    **Observations:**
    - The distribution of ratings per user and per movie shows a long-tail pattern, which is typical for recommendation datasets. A few users rate many movies, and a few movies receive many ratings.
    - Average ratings across different age groups are quite stable, hovering around 3.5.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Correlation Matrix")
        st.pyplot(artifacts['correlation_fig'])
        st.markdown("**Observation:** There are no strong linear correlations between age, rating, and timestamp.")
    
    with col2:
        st.subheader("SVD Explained Variance")
        st.pyplot(artifacts['svd_variance_fig'])
        st.markdown("**Observation:** The first few SVD components capture the most variance, with diminishing returns for additional components.")


# REPLACE the old function with this new one
def show_performance_page():
    st.title("🏆 Model Performance Evaluation")
    st.write("Here we compare the performance of our three core recommendation models based on standard industry metrics.")

    st.subheader("Performance Summary")
    
    # --- Find the best model for each metric ---
    best_rmse_model = results_df.loc[results_df['RMSE'].idxmin()]
    best_precision_model = results_df.loc[results_df['Precision@10'].idxmax()]
    best_recall_model = results_df.loc[results_df['Recall@10'].idxmax()]

    # --- Display Models in a Grid ---
    st.markdown('<div class="performance-grid">', unsafe_allow_html=True)
    
    for index, row in results_df.iterrows():
        model_name = row['Model']
        
        # Determine which metric is the best for this model
        rmse_class = "best-value" if model_name == best_rmse_model['Model'] else ""
        precision_class = "best-value" if model_name == best_precision_model['Model'] else ""
        recall_class = "best-value" if model_name == best_recall_model['Model'] else ""

        st.markdown(f"""
        <div class="model-card">
            <h3 class="model-title">{model_name}</h3>
            <div class="metrics-container">
                <div class="metric-box">
                    <p class="metric-name">RMSE</p>
                    <p class="metric-value {rmse_class}">{row['RMSE']:.4f}</p>
                </div>
                <div class="metric-box">
                    <p class="metric-name">Precision@10</p>
                    <p class="metric-value {precision_class}">{row['Precision@10']:.4f}</p>
                </div>
                <div class="metric-box">
                    <p class="metric-name">Recall@10</p>
                    <p class="metric-value {recall_class}">{row['Recall@10']:.4f}</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    # --- Display Key Takeaways in a styled container ---
    st.subheader("Interpreting the Results")
    st.markdown("""
    <div class="takeaway-container">
        <div class="takeaway-item">
            <i class="bi bi-graph-down-arrow"></i>
            <span><b>RMSE (Lower is Better):</b> Measures prediction error. SVD excels here as it's optimized to minimize this value.</span>
        </div>
        <div class="takeaway-item">
            <i class="bi bi-bullseye"></i>
            <span><b>Precision@10 (Higher is Better):</b> Measures the relevance of the top 10 recommendations. Item-Based models often perform well.</span>
        </div>
        <div class="takeaway-item">
            <i class="bi bi-broadcast"></i>
            <span><b>Recall@10 (Higher is Better):</b> Measures how many of a user's liked items are found in the top 10.</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
# REPLACE the old function with this one
def show_recommender_page():
    st.title("✨ Find Your Next Favorite Movie")
    st.write("") # Add a little vertical space

    # --- User Selection ---
    user_list = sorted(full_data['user_id'].unique())
    selected_user_id = st.selectbox("Choose a User ID:", user_list, index=0)
    num_recs = st.slider("How many recommendations would you like?", 5, 20, 10) # Restored range to 20

    st.divider()

    # --- Display User's Top Rated Movies for Context ---
    st.subheader(f"User {selected_user_id}'s Highest Rated Movies")
    user_ratings = full_data[full_data['user_id'] == selected_user_id]
    top_movies = user_ratings.sort_values('rating', ascending=False).head(5)

    if top_movies.empty:
        st.info("This user has not rated any movies in the dataset.")
    else:
        cols = st.columns(len(top_movies))
        for idx, col in enumerate(cols):
            movie = top_movies.iloc[idx]
            with col:
               st.markdown(f"""
                <div class="history-card">
                <p class="history-rating">{movie['rating']:.0f} ★</p>
                <p class="history-title">{movie['title']}</p>
                </div>
                """, unsafe_allow_html=True)

    st.divider()

    # --- Generate and Display Recommendations in Tabs ---
    st.subheader("Your Personalized Recommendations")
    tab1, tab2, tab3, tab4 = st.tabs(["Hybrid Model", "Item-Based CF", "User-Based CF", "SVD Model"])

    # Helper function to display new V2 cards
    def display_recs_as_cards_v2(recs_df, score_col='predicted_rating', score_name='Predicted Rating'):
        if recs_df.empty:
            st.warning("Could not generate recommendations for this user.")
            return

        # Display recommendations in a single column for the new card format
        for i, row in enumerate(recs_df.head(num_recs).iterrows()):
            movie = row[1]
            st.markdown(f"""
            <div class="rec-card-v2">
                <p class="rec-rank">#{i+1}</p>
                <div class="rec-details-v2">
                    <p class="rec-title-v2">{movie['title']}</p>
                    <p class="rec-score-v2">{score_name}: <b>{movie[score_col]:.2f}</b></p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.write("") # Adds vertical space between cards

    with tab1:
        with st.spinner("Calculating hybrid recommendations..."):
            recs = get_hybrid_recommendations(selected_user_id, n=num_recs)
            display_recs_as_cards_v2(recs, score_col='combined_score', score_name='Match Score')

    with tab2:
        with st.spinner("Calculating item-based recommendations..."):
            recs = get_item_cf_recommendations(selected_user_id, n=num_recs)
            display_recs_as_cards_v2(recs)

    with tab3:
        with st.spinner("Calculating user-based recommendations..."):
            recs = get_user_cf_recommendations(selected_user_id, n=num_recs)
            display_recs_as_cards_v2(recs)

    with tab4:
        with st.spinner("Calculating SVD recommendations..."):
            recs = get_svd_recommendations(selected_user_id, n=num_recs)
            display_recs_as_cards_v2(recs)

# REPLACE the old function with this new one
def show_similar_movies_page():
    st.title("🔍 Find Similar Movies")
    st.write("Choose a movie, and the system will find the most similar ones based on user rating patterns.")

    # --- Widget setup ---
    movie_list = sorted(movies_df['title'].unique())
    selected_movie_title = st.selectbox(
        "Select a movie you like:",
        movie_list,
        index=movie_list.index("Toy Story (1995)") # A good default
    )
    
    num_similar = st.slider("How many similar movies to find?", 5, 20, 10)

    # --- Button to trigger search ---
    if st.button("Find Similar Movies", type="primary"):
        try:
            selected_movie_id = movies_df[movies_df['title'] == selected_movie_title]['movie_id'].iloc[0]
            
            # Get similar movies from the item similarity dataframe
            similarities = item_similarity_df[selected_movie_id].sort_values(ascending=False)[1:num_similar+1]
            similar_movies_df = pd.DataFrame({
                'movie_id': similarities.index,
                'similarity': similarities.values
            }).merge(movies_df, on='movie_id')

            st.divider()
            st.subheader(f"Movies similar to '{selected_movie_title}':")
            st.write("") # Add a little space

            # --- Display results using the V2 cards ---
            if similar_movies_df.empty:
                st.warning("Could not find any similar movies for this selection.")
            else:
                for i, row in enumerate(similar_movies_df.iterrows()):
                    movie = row[1]
                    st.markdown(f"""
                    <div class="rec-card-v2">
                        <p class="rec-rank">#{i+1}</p>
                        <div class="rec-details-v2">
                            <p class="rec-title-v2">{movie['title']}</p>
                            <p class="rec-score-v2">Similarity Score: <b>{movie['similarity']:.2f}</b></p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.write("") # Adds vertical space between cards

        except IndexError:
            st.error("Could not find the selected movie in the dataset.")

# --- SIDEBAR & NAVIGATION (NEW and IMPROVED) ---
with st.sidebar:
    # Add a textual logo
    st.markdown("<div class='sidebar-logo'>FLIX-AI</div>", unsafe_allow_html=True)

      # ADD THIS LINE to load Bootstrap Icons
    st.markdown("""<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">""", unsafe_allow_html=True)
    
    # Get the current page from the query parameters
    if 'page' not in st.query_params:
        current_page_query = "Welcome"
    else:
        current_page_query = st.query_params['page']

    # Define all pages
    pages = {
    "Welcome": '<i class="bi bi-house-door"></i>',
    "EDA": '<i class="bi bi-graph-up-arrow"></i>',
    "Performance": '<i class="bi bi-trophy"></i>',
    "Recommendations": '<i class="bi bi-stars"></i>',
    "Similar Movies": '<i class="bi bi-film"></i>'
    }
    
    # Display the navigation links
    st.markdown("<div class='nav-container'>", unsafe_allow_html=True)
    for page_query, icon in pages.items():
        # Clean page name for display
        page_display_name = page_query.replace("_", " ")
        
        # Check if it's the active page
        is_active = "active" if current_page_query == page_query else ""
        
        # Create the link with an icon
        st.markdown(
    f'<a href="?page={page_query}" target="_self" class="nav-link {is_active}">{icon} {page_display_name}</a>',
    unsafe_allow_html=True
)
    st.markdown("</div>", unsafe_allow_html=True)

# --- Page Routing ---
page_map = {
    "Welcome": show_welcome_page,
    "EDA": show_eda_page,
    "Performance": show_performance_page,
    "Recommendations": show_recommender_page,
    "Similar Movies": show_similar_movies_page
}

# Get the current page from query params, default to Welcome
current_page_key = st.query_params.get("page", "Welcome")
page_to_show = page_map.get(current_page_key, show_welcome_page)
page_to_show()