# 🎵 Spotify Popularity Prediction  
Predict a song’s popularity using its audio features with Machine Learning and a Streamlit web app.

<img src="images/app_screenshot.png" width="800">

---

## 🚀 Project Overview  
This project predicts the **popularity score** of a song using its technical audio features such as:

- Energy  
- Danceability  
- Valence  
- Loudness  
- Acousticness  
- Speechiness  
- Liveness  
- Tempo  

The ML model analyzes these features and estimates how popular a track is likely to be.  
This approach is similar to how Spotify and other music platforms perform **music analytics** and **hit-prediction** modeling.

---

## 🎯 Problem Statement  
Given a song’s audio feature values,  
**can we predict whether the song will be popular or not?**

The model solves this by learning relationships between audio characteristics and Spotify popularity scores.

---

## 🧠 Machine Learning Approach  

### **1. Data Preprocessing**
- Drop irrelevant columns  
- Handle numerical features  
- Standardize using `StandardScaler`  

### **2. Exploratory Data Analysis**
- Scatterplots  
- Correlation heatmap  
- Feature distributions  

### **3. Model Building**
- Algorithm: `RandomForestRegressor`  
- Hyperparameter tuning using `GridSearchCV`  
- Train-test split (80/20)  
- Best estimator saved as a **pickle bundle**  

### **4. Model Deployment**
- UI built using **Streamlit**  
- Loads model bundle (`spotify_rf_bundle.pkl`)  
- Takes user inputs  
- Outputs predicted popularity  

---

## 📂 Project Structure  

