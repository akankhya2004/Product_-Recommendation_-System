🛍️ Product Recommendation System
A machine learning-based application built with Python, Pandas, Scikit-learn, and Streamlit that recommends similar products using product information.

📌 Overview
The system analyzes product category and description, applies TF-IDF vectorization, and uses Cosine Similarity to generate relevant recommendations.

✨ Features
Product-based recommendations
TF-IDF text vectorization
Cosine similarity
Interactive Streamlit interface
CSV export of recommendations
🛠️ Tech Stack
Python
Pandas
Scikit-learn
Streamlit

📂 Project Structure
Product_Recommendation_System/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   ├── products.csv
│   └── ratings.csv
└── model/
    └── recommender.py

⚙️ How It Works
Load product and rating data.
Combine product category and description.
Convert text into TF-IDF vectors.
Calculate Cosine Similarity.
Generate similar product recommendations.
Display results through Streamlit.
Export recommendations as CSV.

🚀 Run Locally
pip install -r requirements.txt
streamlit run app.py

🔮 Future Improvements
Collaborative filtering
Hybrid recommendation system
Personalized recommendations
Product images and filters
Online deployment

👩‍💻 Author
Akankhya