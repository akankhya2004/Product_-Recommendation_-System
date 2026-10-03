import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =====================================================
# 1. CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="AI Product Recommendation System",
    page_icon="🛍️",
    layout="wide"
)

DATA_DIR = "data"
PRODUCTS_FILE = os.path.join(DATA_DIR, "products.csv")
RATINGS_FILE = os.path.join(DATA_DIR, "ratings.csv")

os.makedirs(DATA_DIR, exist_ok=True)


# =====================================================
# 2. SAMPLE DATASET GENERATION
# =====================================================

def create_sample_data():

    if not os.path.exists(PRODUCTS_FILE):

        products = [
            ["P001", "Wireless Headphones", "Electronics",
             "Bluetooth wireless headphones with noise cancellation", 2499],

            ["P002", "Gaming Mouse", "Electronics",
             "RGB gaming mouse with adjustable DPI", 899],

            ["P003", "Mechanical Keyboard", "Electronics",
             "Mechanical keyboard with RGB backlighting", 2499],

            ["P004", "Smart Watch", "Electronics",
             "Fitness smartwatch with heart rate tracking", 3999],

            ["P005", "Bluetooth Speaker", "Electronics",
             "Portable wireless speaker with deep bass", 1799],

            ["P006", "Running Shoes", "Fashion",
             "Lightweight sports shoes for running and fitness", 1999],

            ["P007", "Casual Sneakers", "Fashion",
             "Comfortable everyday casual sneakers", 1599],

            ["P008", "Backpack", "Fashion",
             "Water resistant laptop backpack for college and travel", 1299],

            ["P009", "Smartphone", "Electronics",
             "Android smartphone with powerful processor and camera", 15999],

            ["P010", "Laptop", "Electronics",
             "High performance laptop for programming and productivity", 55999],

            ["P011", "Python Programming Book", "Books",
             "Beginner friendly Python programming and coding book", 599],

            ["P012", "Machine Learning Book", "Books",
             "Machine learning algorithms and artificial intelligence textbook", 899],

            ["P013", "Data Science Course", "Education",
             "Online data science course covering Python pandas and machine learning", 1499],

            ["P014", "Web Development Course", "Education",
             "Full stack web development course with HTML CSS and JavaScript", 1299],

            ["P015", "Yoga Mat", "Fitness",
             "Non slip exercise yoga mat for home workouts", 699],

            ["P016", "Dumbbell Set", "Fitness",
             "Adjustable dumbbell set for strength training", 2499],

            ["P017", "Water Bottle", "Lifestyle",
             "Insulated stainless steel reusable water bottle", 499],

            ["P018", "Desk Lamp", "Lifestyle",
             "LED study desk lamp with adjustable brightness", 899],

            ["P019", "Gaming Headset", "Electronics",
             "Wired gaming headset with microphone and surround sound", 1999],

            ["P020", "Travel Trolley Bag", "Fashion",
             "Lightweight travel luggage trolley bag", 2999]
        ]

        df = pd.DataFrame(
            products,
            columns=[
                "product_id",
                "product_name",
                "category",
                "description",
                "price"
            ]
        )

        df.to_csv(PRODUCTS_FILE, index=False)

    if not os.path.exists(RATINGS_FILE):

        products_df = pd.read_csv(PRODUCTS_FILE)

        rng = np.random.default_rng(42)

        categories = products_df["category"].unique()

        users = [f"U{i:03d}" for i in range(1, 31)]

        ratings = []

        for user in users:

            preferred_categories = rng.choice(
                categories,
                size=2,
                replace=False
            )

            sampled_products = products_df.sample(
                n=10,
                random_state=int(user[1:])
            )

            for _, product in sampled_products.iterrows():

                if product["category"] in preferred_categories:
                    rating = int(rng.choice([3, 4, 5], p=[0.2, 0.4, 0.4]))
                else:
                    rating = int(rng.choice([1, 2, 3, 4, 5]))

                ratings.append([
                    user,
                    product["product_id"],
                    rating
                ])

        ratings_df = pd.DataFrame(
            ratings,
            columns=["user_id", "product_id", "rating"]
        )

        ratings_df.to_csv(RATINGS_FILE, index=False)


create_sample_data()


# =====================================================
# 3. LOAD DATA
# =====================================================

@st.cache_data
def load_data():

    products = pd.read_csv(PRODUCTS_FILE)
    ratings = pd.read_csv(RATINGS_FILE)

    ratings["rating"] = pd.to_numeric(
        ratings["rating"],
        errors="coerce"
    )

    ratings = ratings.dropna(
        subset=["user_id", "product_id", "rating"]
    )

    ratings["rating"] = ratings["rating"].clip(1, 5)

    return products, ratings


products, ratings = load_data()


# =====================================================
# 4. CONTENT-BASED FILTERING
# =====================================================

@st.cache_data
def build_content_similarity(products):

    products = products.copy()

    products["combined_features"] = (
        products["product_name"].fillna("") + " " +
        products["category"].fillna("") + " " +
        products["description"].fillna("")
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    tfidf_matrix = vectorizer.fit_transform(
        products["combined_features"]
    )

    similarity_matrix = cosine_similarity(tfidf_matrix)

    return similarity_matrix


content_similarity = build_content_similarity(products)


def recommend_similar_products(product_id, top_n=5):

    product_indices = products.index[
        products["product_id"] == product_id
    ].tolist()

    if not product_indices:
        return pd.DataFrame()

    index = product_indices[0]

    scores = list(enumerate(content_similarity[index]))

    scores = sorted(
        scores,
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []

    for product_index, score in scores:

        if product_index == index:
            continue

        product = products.iloc[product_index].to_dict()

        product["recommendation_score"] = round(
            float(score) * 100,
            2
        )

        recommendations.append(product)

        if len(recommendations) >= top_n:
            break

    return pd.DataFrame(recommendations)


# =====================================================
# 5. COLLABORATIVE FILTERING
# =====================================================

@st.cache_data
def build_user_item_matrix(ratings):

    return ratings.pivot_table(
        index="user_id",
        columns="product_id",
        values="rating",
        fill_value=0
    )


user_item_matrix = build_user_item_matrix(ratings)


def recommend_for_user(user_id, top_n=5):

    if user_id not in user_item_matrix.index:
        return pd.DataFrame()

    user_ratings = user_item_matrix.loc[user_id]

    # Products already rated by this user
    rated_products = user_ratings[
        user_ratings > 0
    ].index.tolist()

    # Calculate item-item similarity
    item_similarity = cosine_similarity(
        user_item_matrix.T
    )

    item_similarity_df = pd.DataFrame(
        item_similarity,
        index=user_item_matrix.columns,
        columns=user_item_matrix.columns
    )

    recommendations = []

    for product_id in user_item_matrix.columns:

        if product_id in rated_products:
            continue

        similarities = item_similarity_df[product_id]

        rated_similarities = similarities[
            user_ratings > 0
        ]

        if rated_similarities.empty:
            continue

        rated_values = user_ratings[
            rated_similarities.index
        ]

        denominator = rated_similarities.abs().sum()

        if denominator == 0:
            continue

        predicted_rating = (
            rated_similarities * rated_values
        ).sum() / denominator

        recommendations.append({
            "product_id": product_id,
            "predicted_rating": float(predicted_rating)
        })

    if not recommendations:
        return pd.DataFrame()

    result = pd.DataFrame(recommendations)

    result = result.merge(
        products,
        on="product_id",
        how="left"
    )

    result = result.sort_values(
        "predicted_rating",
        ascending=False
    )

    return result.head(top_n)


# =====================================================
# 6. HYBRID RECOMMENDATION SYSTEM
# =====================================================

def recommend_hybrid(user_id, top_n=5):

    collaborative = recommend_for_user(
        user_id,
        top_n=len(products)
    )

    if collaborative.empty:
        return pd.DataFrame()

    # Normalize collaborative predictions to 0-1
    min_rating = collaborative["predicted_rating"].min()
    max_rating = collaborative["predicted_rating"].max()

    if max_rating == min_rating:
        collaborative["collab_score"] = 0.5
    else:
        collaborative["collab_score"] = (
            collaborative["predicted_rating"] - min_rating
        ) / (max_rating - min_rating)

    # Calculate content similarity to user's positively rated products
    user_ratings = user_item_matrix.loc[user_id]

    liked_products = user_ratings[
        user_ratings >= 4
    ].index.tolist()

    content_scores = {}

    for product_id in collaborative["product_id"]:

        target_index = products.index[
            products["product_id"] == product_id
        ][0]

        similarities = []

        for liked_id in liked_products:

            liked_index = products.index[
                products["product_id"] == liked_id
            ]

            if len(liked_index) == 0:
                continue

            similarities.append(
                content_similarity[
                    target_index,
                    liked_index[0]
                ]
            )

        content_scores[product_id] = (
            float(np.mean(similarities))
            if similarities else 0.0
        )

    collaborative["content_score"] = (
        collaborative["product_id"].map(content_scores)
    )

    collaborative["hybrid_score"] = (
        0.6 * collaborative["collab_score"] +
        0.4 * collaborative["content_score"]
    )

    collaborative["recommendation_score"] = (
        collaborative["hybrid_score"] * 100
    ).round(2)

    return collaborative.sort_values(
        "hybrid_score",
        ascending=False
    ).head(top_n)


# =====================================================
# 7. ADD USER RATING
# =====================================================

def save_rating(user_id, product_id, rating_value):

    current_ratings = pd.read_csv(RATINGS_FILE)

    existing = (
        (current_ratings["user_id"] == user_id) &
        (current_ratings["product_id"] == product_id)
    )

    if existing.any():

        current_ratings.loc[
            existing, "rating"
        ] = rating_value

    else:

        new_rating = pd.DataFrame([{
            "user_id": user_id,
            "product_id": product_id,
            "rating": rating_value
        }])

        current_ratings = pd.concat(
            [current_ratings, new_rating],
            ignore_index=True
        )

    current_ratings.to_csv(
        RATINGS_FILE,
        index=False
    )

    st.cache_data.clear()


# =====================================================
# 8. APPLICATION INTERFACE
# =====================================================

st.title("🛍️ AI Product Recommendation System")

st.markdown(
    """
    Discover personalized products using machine learning,
    content-based filtering, and collaborative filtering.
    """
)

st.divider()


# Sidebar
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose a section",
    [
        "Home",
        "Product Explorer",
        "Content-Based Recommendations",
        "Collaborative Recommendations",
        "Hybrid Recommendations",
        "Rate Products",
        "Analytics"
    ]
)


# =====================================================
# HOME
# =====================================================

if page == "Home":

    st.header("Recommendation Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Products",
        len(products)
    )

    col2.metric(
        "Total Users",
        ratings["user_id"].nunique()
    )

    col3.metric(
        "Total Ratings",
        len(ratings)
    )

    st.subheader("Available Categories")

    categories = products["category"].unique()

    cols = st.columns(3)

    for i, category in enumerate(categories):
        cols[i % 3].info(category)

    st.subheader("Featured Products")

    featured = products.sample(
        min(6, len(products)),
        random_state=10
    )

    cols = st.columns(3)

    for i, (_, product) in enumerate(featured.iterrows()):

        with cols[i % 3]:

            st.markdown(f"### {product['product_name']}")

            st.caption(product["category"])

            st.write(product["description"])

            st.write(f"Price: ₹{product['price']:,.0f}")


# =====================================================
# PRODUCT EXPLORER
# =====================================================

elif page == "Product Explorer":

    st.header("Explore Products")

    category = st.selectbox(
        "Filter by category",
        ["All"] + sorted(products["category"].unique())
    )

    search = st.text_input(
        "Search products",
        placeholder="Enter product name..."
    )

    filtered = products.copy()

    if category != "All":
        filtered = filtered[
            filtered["category"] == category
        ]

    if search.strip():
        filtered = filtered[
            filtered["product_name"].str.contains(
                search,
                case=False,
                na=False
            )
        ]

    st.write(f"Found {len(filtered)} products")

    for _, product in filtered.iterrows():

        with st.container(border=True):

            st.subheader(product["product_name"])

            st.write(product["description"])

            st.write(f"Category: {product['category']}")

            st.write(f"Price: ₹{product['price']:,.0f}")

            st.caption(f"Product ID: {product['product_id']}")


# =====================================================
# CONTENT-BASED RECOMMENDATIONS
# =====================================================

elif page == "Content-Based Recommendations":

    st.header("Content-Based Product Recommendations")

    st.write(
        "Find products similar to a product you already like."
    )

    selected_product = st.selectbox(
        "Choose a product",
        products["product_id"],
        format_func=lambda x: products.loc[
            products["product_id"] == x,
            "product_name"
        ].iloc[0]
    )

    top_n = st.slider(
        "Number of recommendations",
        1, 10, 5
    )

    if st.button("Find Similar Products"):

        result = recommend_similar_products(
            selected_product,
            top_n
        )

        st.subheader("Recommended Products")

        st.dataframe(
            result[
                [
                    "product_name",
                    "category",
                    "price",
                    "recommendation_score"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download Recommendations CSV",
            result.to_csv(index=False),
            "content_recommendations.csv",
            "text/csv"
        )


# =====================================================
# COLLABORATIVE RECOMMENDATIONS
# =====================================================

elif page == "Collaborative Recommendations":

    st.header("Collaborative Filtering")

    st.write(
        "Recommendations are based on patterns in user ratings."
    )

    users = sorted(user_item_matrix.index.tolist())

    selected_user = st.selectbox(
        "Select user",
        users
    )

    top_n = st.slider(
        "Number of recommendations",
        1, 10, 5
    )

    if st.button("Recommend Products"):

        result = recommend_for_user(
            selected_user,
            top_n
        )

        if result.empty:
            st.warning(
                "Not enough rating information to generate recommendations."
            )

        else:

            st.dataframe(
                result[
                    [
                        "product_name",
                        "category",
                        "price",
                        "predicted_rating"
                    ]
                ],
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "Download Recommendations CSV",
                result.to_csv(index=False),
                "collaborative_recommendations.csv",
                "text/csv"
            )


# =====================================================
# HYBRID RECOMMENDATIONS
# =====================================================

elif page == "Hybrid Recommendations":

    st.header("Hybrid Recommendation Engine")

    st.write(
        "Combines collaborative filtering and content similarity."
    )

    selected_user = st.selectbox(
        "Select user",
        sorted(user_item_matrix.index.tolist()),
        key="hybrid_user"
    )

    top_n = st.slider(
        "Number of recommendations",
        1, 10, 5,
        key="hybrid_top_n"
    )

    if st.button("Generate Hybrid Recommendations"):

        result = recommend_hybrid(
            selected_user,
            top_n
        )

        if result.empty:

            st.warning("Unable to generate recommendations.")

        else:

            st.subheader("Personalized Recommendations")

            for _, product in result.iterrows():

                with st.container(border=True):

                    st.subheader(product["product_name"])

                    st.write(product["description"])

                    st.write(
                        f"Category: {product['category']}"
                    )

                    st.write(
                        f"Price: ₹{product['price']:,.0f}"
                    )

                    st.progress(
                        float(product["recommendation_score"]) / 100
                    )

                    st.caption(
                        f"Hybrid recommendation score: "
                        f"{product['recommendation_score']}%"
                    )

            st.download_button(
                "Export Hybrid Recommendations",
                result.to_csv(index=False),
                "hybrid_recommendations.csv",
                "text/csv"
            )


# =====================================================
# RATE PRODUCTS
# =====================================================

elif page == "Rate Products":

    st.header("Rate Products")

    st.write(
        "Submit ratings to update the recommendation dataset."
    )

    user_id = st.selectbox(
        "Select user",
        sorted(ratings["user_id"].unique()),
        key="rating_user"
    )

    product_id = st.selectbox(
        "Select product",
        products["product_id"],
        format_func=lambda x: products.loc[
            products["product_id"] == x,
            "product_name"
        ].iloc[0],
        key="rating_product"
    )

    rating_value = st.slider(
        "Your rating",
        1, 5, 5
    )

    if st.button("Submit Rating"):

        save_rating(
            user_id,
            product_id,
            rating_value
        )

        st.success("Rating saved successfully!")

        st.rerun()

    st.subheader("Existing Ratings")

    user_history = ratings[
        ratings["user_id"] == user_id
    ].merge(
        products,
        on="product_id",
        how="left"
    )

    st.dataframe(
        user_history[
            ["product_name", "category", "rating"]
        ],
        use_container_width=True,
        hide_index=True
    )


# =====================================================
# ANALYTICS
# =====================================================

elif page == "Analytics":

    st.header("Dataset Analytics")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Products by Category")

        category_counts = products["category"].value_counts()

        fig = px.bar(
            x=category_counts.index,
            y=category_counts.values,
            labels={
                "x": "Category",
                "y": "Number of Products"
            },
            title="Product Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        st.subheader("Rating Distribution")

        fig = px.histogram(
            ratings,
            x="rating",
            nbins=5,
            title="User Ratings"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.subheader("Most Rated Products")

    rating_counts = ratings.groupby(
        "product_id"
    ).size().reset_index(name="rating_count")

    popular_products = rating_counts.merge(
        products,
        on="product_id"
    ).sort_values(
        "rating_count",
        ascending=False
    )

    fig = px.bar(
        popular_products.head(10),
        x="product_name",
        y="rating_count",
        title="Top 10 Most Rated Products"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Average Product Ratings")

    average_ratings = ratings.groupby(
        "product_id"
    )["rating"].mean().reset_index()

    average_ratings = average_ratings.merge(
        products,
        on="product_id"
    )

    st.dataframe(
        average_ratings[
            ["product_name", "category", "rating"]
        ].sort_values(
            "rating",
            ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "AI-Based Product Recommendation System | "
    "Python • Pandas • Scikit-learn • Streamlit"
)