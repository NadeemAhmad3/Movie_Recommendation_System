# 🎬 Flix-AI: A Hybrid Movie Recommender System

![python-shield](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![pandas-shield](https://img.shields.io/badge/pandas-2.2-blue)
![sklearn-shield](https://img.shields.io/badge/scikit--learn-1.6-blue)
![xgboost-shield](https://img.shields.io/badge/XGBoost-2.1-blue)
![streamlit-shield](https://img.shields.io/badge/Streamlit-1.3-ff69b4)

A **sophisticated machine learning project** that builds and evaluates multiple movie recommendation algorithms using the MovieLens dataset. This repository contains the complete workflow—from exploratory data analysis (EDA) and model implementation in a Jupyter Notebook to a polished, interactive web application built with Streamlit.

> 💡 The project culminates in a **Hybrid Recommender** that intelligently combines the strengths of User-Based Collaborative Filtering, Item-Based Collaborative Filtering, and SVD models to provide robust and diversified movie suggestions.


---

## 🌟 Features

- ✨ **Hybrid Recommendations**: Get movie suggestions from a weighted model that combines User-CF, Item-CF, and SVD predictions.
- 🎯 **Multiple Models**: Compare recommendations from four different algorithms side-by-side.
- 🔍 **Find Similar Movies**: Select a movie and instantly discover others that are most similar based on user rating patterns.
- 📊 **In-Depth EDA**: An entire section dedicated to visualizing user demographics, rating behaviors, and data distributions.
- 🏆 **Performance Dashboard**: Interactively view and compare the performance metrics (RMSE, Precision@10, Recall@10) for each model.
- 🎨 **Modern UI**: A clean, intuitive, and visually appealing user interface with custom styling.

---

## 🛠️ Tech Stack

| Category              | Tools & Libraries                                 |
|-----------------------|---------------------------------------------------|
| **Data Processing**   | Pandas, NumPy SciPy                               |
| **Visualization**     | Matplotlib, Seaborn                               |
| **Machine Learning**  | Scikit-learn (SVD, Cosine Similarity, Metrics)    |
| **Web App**           | Streamlit                                         |
| **Dev Environment**   | JupyterLab, Python 3.9+                           |
| **Dev Persistence**   | Pickle                                            |

---

## 📁 Project Structure

```bash
.
├── input/
│   ├── u.data
│   ├── u.item
│   ├── u.user
│   └── (and other )
├── saved_models/
│   ├── movies.pkl
│   ├── train_matrix.pkl
│   ├── user_similarity_df.pkl
│   └── (and other )
├── .ipynb    # Jupyter Notebook for model training
├── app.py                  # Main Streamlit web app
├── style.css                         # Custom CSS styling
├── requirements.txt                  # Dependencies

```

## ⚙️ Installation & Setup
**1. Clone the Repository** 
```bash
git clone https://github.com/NadeemAhmad3/Movie_Recommendation_System.git
cd your-repo-name
```
**2. Create a Virtual Environment (Recommended)**
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix/Mac
source venv/bin/activate
```
**3. Install Dependencies**
**Method 1:** From requirements.txt
```bash
pip install -r requirements.txt
```
**Method 2:** Manual Installation
```bash
# Core Data Stack
pip install pandas numpy scipy

# Visualization
pip install matplotlib seaborn 

# ML & Modeling
pip install scikit-learn

# Jupyter Environment
pip install jupyterlab ipykernel

```
## ▶️ How to Run the Project
**1. Run the Jupyter Notebook** 
```bash
jupyter lab
```
Then open .ipynb to walk through data analysis and model training.

**2. Launch the Streamlit App** 
```bash
streamlit run streamlit_app.py
```
Your browser will automatically open the app at http://localhost:8501.

## 🧠 Modeling & Results

Three primary recommendation models were built and evaluated, which are then combined into a hybrid system.

| Model         | RMSE *(Lower is Better)* | Precision@10 *(Higher is Better)*  | Recall@10 *(Higher is Better)* | Notes                                               |
| ------------- | ------------------------ | ---------------------------------- | ------------------------------ | --------------------------------------------------- |
| **SVD**       | **0.9997**               | 0.0381                             | 0.0242                         | Best for prediction accuracy (lowest error).        |
| Item-Based CF | 1.0269                   | **0.0528**                         | **0.0321**                     | Best for relevance of recommendations.              |
| User-Based CF | 1.0375                   | 0.0467                             | 0.0276                         | A classic collaborative filtering approach.         |

✅ The final **Hybrid Model** leverages the strengths of all three, providing a balanced and effective recommendation experience to the end-user.


## 🤝 Contributing
We welcome contributions!

**1.** Fork the repo

**2.** Create your feature branch
```bash
git checkout -b feature/AmazingFeature
```
**3.** Commit your changes
```bash
git commit -m "Add some AmazingFeature"
```
**4.** Push to your branch
```bash
git push origin feature/AmazingFeature
```
**5.** Open a Pull Request

## 📧 Contact
**Nadeem Ahmad**

📫 **onedaysuccussfull@gmail.com**
