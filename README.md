🎬 Flix-AI: A Hybrid Movie Recommender System

![alt text](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![alt text](https://img.shields.io/badge/pandas-2.x-blue)
![alt text](https://img.shields.io/badge/scikit--learn-1.x-blue)
![alt text](https://img.shields.io/badge/Seaborn-0.12-purple)
![alt text](https://img.shields.io/badge/Streamlit-1.x-ff69b4)

An in-depth data science project that builds and evaluates multiple movie recommendation algorithms using the MovieLens 100k dataset. This repository contains the complete workflow—from exploratory data analysis and model implementation in a Jupyter Notebook to a polished, interactive web application built with Streamlit.

💡 The project culminates in a Hybrid Recommender that intelligently combines the strengths of User-Based Collaborative Filtering, Item-Based Collaborative Filtering, and SVD models to provide robust and diversified movie suggestions.

🌟 Features

✨ Hybrid Recommendations: Get movie suggestions from a weighted model that combines User-CF, Item-CF, and SVD predictions.

🎯 Multiple Models: Compare recommendations from four different algorithms side-by-side.

🔍 Find Similar Movies: Select a movie and instantly discover others that are most similar based on user rating patterns.

📊 In-Depth EDA: An entire section dedicated to visualizing user demographics, rating behaviors, and data distributions.

🏆 Performance Dashboard: Interactively view and compare the performance metrics (RMSE, Precision@10, Recall@10) for each model.

🎨 Modern UI: A clean, intuitive, and visually appealing user interface with custom styling.

🛠️ Tech Stack
Category	Tools & Libraries
Data Processing	Pandas, NumPy, SciPy
Visualization	Matplotlib, Seaborn
Machine Learning	Scikit-learn (SVD, Cosine Similarity, Metrics)
Web App	Streamlit
Data Persistence	Pickle
Dev Environment	JupyterLab, Python 3.9+
📁 Project Structure
code
Bash
download
content_copy
expand_less

.
├── input/                  # MovieLens 100k dataset files
│   ├── u.data
│   ├── u.item
│   └── u.user
├── saved_models/           # Directory for all pickled models and data
│   ├── movies.pkl
│   ├── train_matrix.pkl
│   ├── user_similarity_df.pkl
│   ├── item_similarity_df.pkl
│   ├── svd_model.pkl
│   ├── results_df.pkl
│   └── ... (and other artifacts)
├── recommendation_notebook.ipynb # Notebook for EDA, model training & evaluation
├── app.py                  # Main Streamlit web application
├── style.css               # Custom CSS for the Streamlit app
└── requirements.txt        # Project dependencies
⚙️ Installation & Setup

1. Clone the Repository

code
Bash
download
content_copy
expand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
git clone https://github.com/your-username/Flix-AI.git
cd Flix-AI

2. Create a Virtual Environment (Recommended)

code
Bash
download
content_copy
expand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix/Mac
source venv/bin/activate

3. Install Dependencies
Method 1: From requirements.txt

code
Bash
download
content_copy
expand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
pip install -r requirements.txt

Method 2: Manual Installation```bash

Core Data Stack

pip install pandas numpy scipy

Visualization

pip install matplotlib seaborn

ML & Modeling

pip install scikit-learn

Web App

pip install streamlit

Jupyter Environment

pip install jupyterlab ipykernel

code
Code
download
content_copy
expand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
## ▶️ How to Run the Project
This is a two-step process. You **must run the notebook first** to generate the models and data artifacts that the Streamlit app depends on.

**1. Run the Jupyter Notebook** 
First, launch JupyterLab to run the analysis and model training notebook. This will create and populate the `saved_models/` directory.
```bash
jupyter lab

Inside JupyterLab, open recommendation_notebook.ipynb and run all the cells from top to bottom.

2. Launch the Streamlit App
Once the notebook has been fully executed, you can launch the interactive web application.

code
Bash
download
content_copy
expand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
streamlit run app.py

Your browser will automatically open the app at http://localhost:8501.

🧠 Modeling & Results

Three primary recommendation models were built and evaluated.

Model	RMSE (Lower is Better)	Precision@10 (Higher is Better)	Recall@10 (Higher is Better)	Notes
SVD	0.9997	0.0381	0.0242	Best for prediction accuracy (lowest error).
Item-Based CF	1.0269	0.0528	0.0321	Best for relevance of top recommendations.
User-Based CF	1.0375	0.0467	0.0276	A classic collaborative filtering approach.

✅ The final deployed model is a Hybrid System that combines the normalized scores from all three models, leveraging their individual strengths.

🤝 Contributing

We welcome contributions!

1. Fork the repo

2. Create your feature branch

code
Bash
download
content_copy
expand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
git checkout -b feature/AmazingFeature

3. Commit your changes

code
Bash
download
content_copy
expand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
git commit -m "Add some AmazingFeature"

4. Push to your branch

code
Bash
download
content_copy
expand_less
IGNORE_WHEN_COPYING_START
IGNORE_WHEN_COPYING_END
git push origin feature/AmazingFeature

5. Open a Pull Request

📧 Contact

Nadeem Ahmad

📫 onedaysuccussfull@gmail.com
