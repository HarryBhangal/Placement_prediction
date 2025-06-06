# 🎓 WHILE YOU BE PLACED

A simple machine learning web app built using **Streamlit** and **Logistic Regression** that predicts whether a student is likely to be placed based on their **Age**, **Number of Internships**, and **CGPA**.

---

## 🚀 Features

- Predicts the placement status of a student using a trained ML model.
- Visualizes feature correlations using a heatmap.
- User-friendly interface built with Streamlit.
- Input fields for Age, Internships, and CGPA.
- Displays prediction results interactively.

---

## 🧠 Tech Stack

- **Python**
- **Pandas** for data manipulation
- **Seaborn & Matplotlib** for data visualization
- **Scikit-learn** for machine learning
- **Streamlit** for the web interface

---

## 📁 Dataset

The dataset used is `collegePlace.csv`, which includes:

- `Age`: Age of the student  
- `Internships`: Number of internships completed  
- `CGPA`: Cumulative GPA  
- `PlacedOrNot`: Target variable (1 = Placed, 0 = Not Placed)

---

## 🧪 Model

We used a **Logistic Regression** model trained on selected features:

```python
X = df[['Age', 'Internships', 'CGPA']]
Y = df['PlacedOrNot']
```

The model is trained using an 80-20 train-test split and evaluated using accuracy score.

---

## 📈 Heatmap Visualization

A heatmap is displayed to show correlation between numeric features, which helps in understanding which factors are most related to placement success.

---

## 💻 How to Run

1. Clone the repository or download the project files.
2. Ensure you have Python installed (preferably 3.8+).
3. Install dependencies:

```bash
pip install pandas scikit-learn seaborn matplotlib streamlit
```

4. Run the Streamlit app:

```bash
streamlit run app.py
```

5. Open your browser to the local URL provided (usually http://localhost:8501).

---

## 🎯 Usage

1. Enter your **Age**, **Number of Internships**, and **CGPA** in the input fields.
2. Click **Predict**.
3. See whether you're likely to be placed: **Yes** or **No**!

---

## 📌 Example

> Age: `21`  
> Internships: `2`  
> CGPA: `8.5`  
> **Result**: ✅ *Yes, you are likely to be placed!*

---

## 📬 Contact

For suggestions or queries, feel free to reach out!

﻿# Placement_prediction
