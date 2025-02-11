# ... [Previous code remains unchanged until the recommend_news function] ...

# 6️⃣ Enhanced Recommendation Function with Hybrid Focus (Corrected)
import scipy.sparse as sp 
import pandas as pd# Add this import at the top of your code
cleaned_data=pd.read_csv('updated.csv')
import streamlit as st
import numpy as np
import scipy.sparse as sp
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load your data (replace with your actual data loading logic)
# cleaned_data = pd.read_csv('your_data.csv')

# 1️⃣ Aggregate News Content with Proper Validation
news_grouped = cleaned_data.groupby('News ID')['Combined'].agg(lambda texts: ' '.join(texts.astype(str))).reset_index()

# Add Main Category to the grouped data
news_grouped = news_grouped.merge(
    cleaned_data[['News ID', 'Main Category']].drop_duplicates(),
    on='News ID',
    how='left'
)

# Validate data
assert news_grouped['News ID'].nunique() == len(cleaned_data['News ID'].unique()), "News ID mismatch in grouping!"
assert news_grouped['Main Category'].isnull().sum() == 0, "Missing Main Categories detected!"

# 2️⃣ Enhanced TF-IDF Vectorization
vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2),
    max_df=0.85,
    min_df=2,
    sublinear_tf=True,
    max_features=10000
)

news_embeddings = vectorizer.fit_transform(news_grouped['Combined'])

# 3️⃣ Efficient Similarity Calculation
content_similarity_matrix = cosine_similarity(news_embeddings, dense_output=False)

# 4️⃣ Category Similarity Matrix
categories = news_grouped['Main Category'].unique()
category_to_idx = {cat: idx for idx, cat in enumerate(categories)}
category_matrix = np.zeros((len(news_grouped), len(categories)))

for idx, row in news_grouped.iterrows():
    category_matrix[idx, category_to_idx[row['Main Category']]] = 1

category_similarity_matrix = cosine_similarity(category_matrix, dense_output=False)

# 5️⃣ Hybrid Similarity Matrix
hybrid_similarity_matrix = 0.6 * content_similarity_matrix + 0.4 * category_similarity_matrix

# ... [Previous code remains unchanged until the recommend_news function] ...

# 6️⃣ Enhanced Recommendation Function with Hybrid Focus (Corrected)
def recommend_news(user_id, raw_df, grouped_df, similarity_matrix, top_n=5):
    news_ids = grouped_df['News ID'].tolist()
    news_to_idx = {news_id: idx for idx, news_id in enumerate(news_ids)}
    
    user_clicked = raw_df[raw_df['User ID'] == user_id]['News ID'].unique()
    if not user_clicked.size:
        st.warning(f"User {user_id} has no click history.")
        return []
    
    scores = np.zeros(len(news_ids))
    valid_click_count = 0
    
    for news_id in user_clicked:
        if news_id in news_to_idx:
            row_index = news_to_idx[news_id]
            row = similarity_matrix[row_index]
            if sp.issparse(row):
                row_scores = row.toarray().flatten()
            else:
                row_scores = np.asarray(row).flatten()
            scores += row_scores
            valid_click_count += 1
    
    if valid_click_count == 0:
        st.warning(f"No valid clicked news found for user {user_id}")
        return []
        
    scores /= valid_click_count  # Normalize by valid clicks
    
    clicked_set = set(user_clicked)
    sorted_indices = np.argsort(scores)[::-1]
    
    recommendations = []
    for idx in sorted_indices:
        if len(recommendations) >= top_n:
            break
        candidate = news_ids[idx]
        if candidate not in clicked_set:
            recommendations.append(candidate)
    
    return recommendations
import streamlit as st
import numpy as np
import scipy.sparse as sp
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# [Previous code remains unchanged until the Streamlit app section]

st.title("Personalized News Recommendation System")

# Get unique user IDs from the cleaned data
unique_users = sorted(cleaned_data['User ID'].unique())

# Create a dropdown selector for User ID
selected_user = st.selectbox(
    "Select User ID",
    options=unique_users,
    index=0,
    format_func=lambda x: f"User {x}"
)

# Button to generate recommendations
if st.button("Get Recommendations"):
    # Display original news articles viewed by the user
    user_history = cleaned_data[cleaned_data['User ID'] == selected_user]['News ID'].unique()
    
    # Create two columns for side-by-side comparison
    col1, col2 = st.columns(2)
    
    # Column 1: Originally viewed news
    with col1:
        st.header("Originally Viewed News")
        if len(user_history) > 0:
            for i, news_id in enumerate(user_history, 1):
                news_info = cleaned_data[cleaned_data['News ID'] == news_id].iloc[0]
                
                with st.expander(f"Article {i}: {news_info['News Title'][:50]}..."):
                    st.write(f"News ID: {news_id}")
                    st.write(f"Title: {news_info['News Title']}")
                    st.write(f"Content: {news_info['News']}")
                    st.write(f"Category: {news_info['Main Category']}")
        else:
            st.warning("No viewing history found for this user.")
    
    # Column 2: Recommended news
    with col2:
        st.header("Recommended News")
        recommendations = recommend_news(
            user_id=selected_user,
            raw_df=cleaned_data,
            grouped_df=news_grouped,
            similarity_matrix=hybrid_similarity_matrix,
            top_n=5
        )
        
        if recommendations:
            for i, news_id in enumerate(recommendations, 1):
                news_info = cleaned_data[cleaned_data['News ID'] == news_id].iloc[0]
                
                with st.expander(f"Article {i}: {news_info['News Title'][:50]}..."):
                    st.write(f"News ID: {news_id}")
                    st.write(f"Title: {news_info['News Title']}")
                    st.write(f"Content: {news_info['News']}")
                    st.write(f"Category: {news_info['Main Category']}")
        else:
            st.warning("No recommendations available for this user.")
    
    # Add similarity analysis
    if recommendations and len(user_history) > 0:
        st.header("Similarity Analysis")
        
        # Calculate category overlap
        original_categories = set(cleaned_data[cleaned_data['News ID'].isin(user_history)]['Main Category'])
        recommended_categories = set(cleaned_data[cleaned_data['News ID'].isin(recommendations)]['Main Category'])
        
        category_overlap = len(original_categories.intersection(recommended_categories))
        
        # Display statistics
        st.write(f"Number of articles viewed: {len(user_history)}")
        st.write(f"Number of recommendations: {len(recommendations)}")
        st.write(f"Category overlap: {category_overlap} categories")
        st.write("Original categories:", ", ".join(original_categories))
        st.write("Recommended categories:", ", ".join(recommended_categories))

