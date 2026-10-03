🛍️ Product Recommendation System

A Product Recommendation System built using Python and Machine Learning that recommends products to users based on their previous ratings and product preferences.

📌 Project Overview

The goal of this project is to provide personalized product recommendations by analyzing user-product ratings. The system uses recommendation techniques to identify products that a user may be interested in.

This project demonstrates the practical implementation of Machine Learning, data processing, and recommendation systems.

🚀 Features

- 📊 Analyze user-product ratings
- 👤 Generate personalized product recommendations
- 🔍 Find similar products
- 🤖 Machine Learning-based recommendation approach
- 🌐 Simple user interface using Streamlit
- 📁 CSV-based dataset for products and ratings

🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Machine Learning

📂 Project Structure

Product_Recommendation_System/
│
├── app.py
├── recommender.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── data/
    ├── products.csv
    └── ratings.csv

📄 File Description

"app.py"

The main Streamlit application that provides the user interface and displays product recommendations.

"recommender.py"

Contains the recommendation logic and functions used to generate product recommendations.

"products.csv"

Contains product information such as product ID, product name, category, and other product details.

"ratings.csv"

Contains user ratings for different products.

"requirements.txt"

Contains the Python libraries required to run the project.

⚙️ Installation

1. Clone the repository

git clone https://github.com/yourusername/Product_Recommendation_System.git

2. Navigate to the project directory

cd Product_Recommendation_System

3. Create a virtual environment

python -m venv venv

Activate it:

Windows:

venv\Scripts\activate

Mac/Linux:

source venv/bin/activate

4. Install dependencies

pip install -r requirements.txt

▶️ Run the Application

Run the Streamlit application using:

streamlit run app.py

The application will open in your browser.

🔄 How It Works

User Ratings
     ↓
Data Processing
     ↓
Recommendation Algorithm
     ↓
Analyze Product Similarity
     ↓
Generate Recommendations
     ↓
Display Recommended Products

The system processes user ratings and product information to identify products that are relevant to the user's preferences.

📊 Example

A user selects or rates a product:

Selected Product: Smartphone

The system analyzes product information and ratings and may recommend:

Recommended Products:
1. Wireless Earbuds
2. Smart Watch
3. Phone Case
4. Power Bank
5. Bluetooth Speaker

🎯 Future Improvements

- Add a larger real-world dataset
- Implement collaborative filtering
- Implement content-based filtering
- Combine multiple recommendation techniques
- Add user authentication
- Deploy the application online
- Add product images and detailed product information
- Improve recommendation accuracy

👩‍💻 Author

Akankhya