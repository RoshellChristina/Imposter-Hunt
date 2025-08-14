Imposter Hunt

A Flask web application that predicts which of two texts is real vs fake using ML/NLP models.

Features

- User registration and login
- Submit text pairs for prediction
- View prediction history
- Uses CatBoost, BERT, TF-IDF, and EDA features for prediction

## Setup

1. Clone the repository to your local machine.
2. Install all required dependencies
3. Make sure the `models/` folder contains all trained models (`.pkl` and `.joblib` files).
4. Run the Flask application:

5. Open your browser and go to `http://127.0.0.1:5000/` to access the app.

## Notes

- NLTK data will be downloaded automatically if not already present.
- Database credentials can be configured in `config.py`.
- Ensure your Python environment has the necessary packages (listed in `requirements.txt`) for smooth operation.


