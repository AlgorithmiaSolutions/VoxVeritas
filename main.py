from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import xgboost as xgb
import librosa
import numpy as np
import pandas as pd
import joblib
import os
import glob
import warnings

def extract_features_from_segment(segment, sr):
    # Extract features from the segment
    chroma = librosa.feature.chroma_stft(y=segment, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=segment, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=segment, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=segment, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=segment)
    mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=20)
    rms = librosa.feature.rms(y=segment)

    # Aggregate features
    features = {
        'chroma_stft': np.mean(chroma),
        'rms': np.mean(rms),
        'spectral_centroid': np.mean(spectral_centroid),
        'spectral_bandwidth': np.mean(spectral_bandwidth),
        'rolloff': np.mean(spectral_rolloff),
        'zero_crossing_rate': np.mean(zero_crossing_rate)
    }

    # Include MFCCs separately
    for i in range(20):
        features[f'mfcc{i + 1}'] = np.mean(mfccs[i])

    return features


def extract_features(file_path):
    # Label: AI = 1; Human = 0
    y, sr = librosa.load(file_path, sr=None)
    df = pd.DataFrame()

    # Apply moving average filter for noise reduction
    window_size = 1000  # Adjust window size as needed
    denoised_audio = np.convolve(y, np.ones(window_size)/window_size, mode='same')

    # Split the audio into 1-second segments
    segment_length = sr  # 1 second worth of samples
    num_segments = len(denoised_audio) // segment_length


    for i in range(num_segments):
        start = i * segment_length
        end = start + segment_length
        segment = denoised_audio[start:end]
        # Extract features from this segment, ignoring segments that would cause warnings due to being too short or otherwise
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            try:
                segment_features = extract_features_from_segment(segment, sr)
            except UserWarning:
                print("Pitch estimation not possible for the given segment. Skipping...")
                continue

        df = df._append(segment_features, ignore_index=True)
    return df

# Data Preprocessing: Preprocess extracted features
file_path = "xgboost_model.pkl"

featuresfile = pd.read_csv('./Samples/KAGGLE/DATASET-balanced.csv')

# Assuming your features are stored in columns (with the last column being the label)
features = featuresfile.drop('label', axis = 1).copy()  # Extract features (all columns except label)
labels = featuresfile['label'].copy()   # Extract labels

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, shuffle=True, random_state=420)
if not os.path.exists(file_path):
    # Making a parameter grid for Hyperparameter Tuning
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
        'min_child_weight': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'reg_lambda': [0.01, 0.1, 1.0, 10.0, 50.0]
    }

    # Initialize the XGBoost classifier
    clf = xgb.XGBClassifier(objective='binary:logistic', eval_metric='aucpr')

    # Perform random search with cross-validation
    random_search = RandomizedSearchCV(clf, param_distributions=param_grid, n_iter=50, scoring='roc_auc', cv=5, verbose=1, n_jobs=2)
    random_search.fit(X_train, y_train)

    # Get the best hyperparameters
    best_params = random_search.best_params_

    # Training an XGBoost Classifier
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train)
    joblib.dump(model, 'xgboost_model.pkl')

model = joblib.load('xgboost_model.pkl') # Use this to load the model
test_result = model.predict(X_test)
threshold = 0.9
test_result_class = [1 if p > threshold else 0 for p in test_result]
print(confusion_matrix(y_test, test_result_class))

# Prediction on new samples
folder_path = ".\Samples\AITest_LE_30s"
new_samples = glob.glob(folder_path + "\*")

folder_size = len(new_samples)
sum_of_guesses = 0
# new_samples = ["Samples\HumanTest_LE_10s\HumanTest_112.mp3"]
for sample in new_samples:
    features = extract_features(sample)
    prediction = model.predict(features)
    prediction = [1 if p > threshold else 0 for p in prediction]
    print(confusion_matrix(y_test, prediction))
    print(f"{sample}: Predicted class - {'AI' if np.mean(prediction) > threshold else 'Human'} - {np.mean(prediction)}\n")
    sum_of_guesses += np.mean(prediction)

# print(f"Accuracy: {model.score(X_test,y_test)}")
print(f"Average Guess: {sum_of_guesses/folder_size}")