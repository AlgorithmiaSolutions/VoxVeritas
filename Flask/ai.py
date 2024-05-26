from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import xgboost as xgb
import librosa
import numpy as np
import pandas as pd
import joblib
import os
import glob

THRESHOLD = 0.9
FOLDER_PATH = 'uploads/'


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


def extract_features_labelled(file_path, label):
    # Label: AI = 1; Human = 0
    y, sr = librosa.load(file_path, sr=None)
    df = pd.DataFrame()

    # Split the audio into 1-second segments
    segment_length = sr  # 1 second worth of samples
    num_segments = len(y) // segment_length

    for i in range(num_segments):
        start = i * segment_length
        end = start + segment_length
        segment = y[start:end]

        # Extract features from this segment
        segment_features = extract_features_from_segment(segment, sr)

        # Add label column (AI = 1; Human = 0)
        segment_features['label'] = label

        df = df._append(segment_features, ignore_index=True)

    return df


def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    df = pd.DataFrame()

    # Split the audio into 1-second segments
    segment_length = sr  # 1 second worth of samples
    num_segments = len(y) // segment_length

    for i in range(num_segments):
        start = i * segment_length
        end = start + segment_length
        segment = y[start:end]

        # Extract features from this segment
        segment_features = extract_features_from_segment(segment, sr)

        df = df._append(segment_features, ignore_index=True)

    return df


# # # # Initialize an empty dataframe to store the features
# df = pd.DataFrame()

# # # # Extracting features from the human voice audio file
# human_files = glob.glob('./KAGGLE/AUDIO/REAL/*.wav')
# for file in human_files:
#     df = df._append(extract_features_labelled(file, 0))

# # # # Extracting features from the AI voice audio file and appending it to dataframe
# ai_files = glob.glob('./KAGGLE/AUDIO/FAKE/*.wav')
# for file in ai_files:
#     df = df._append(extract_features_labelled(file, 1))

# # # # Saving it to a csv file
# df.to_csv('output.csv', index=False)

# Data Preprocessing: Preprocess extracted features
def ai_process(audio_file):
    model_file_path = "xgboost_model.pkl"

    features_file = pd.read_csv('../Samples/KAGGLE/DATASET-balanced.csv')

    # Assuming your features are stored in columns (with the last column being the label)
    features = features_file.drop('label', axis=1).copy()  # Extract features (all columns except label)
    labels = features_file['label'].copy()  # Extract labels

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, shuffle=True, random_state=420)
    if not os.path.exists(model_file_path):
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
        random_search = RandomizedSearchCV(clf, param_distributions=param_grid, n_iter=50, scoring='roc_auc', cv=5,
                                           verbose=1, n_jobs=2)
        random_search.fit(X_train, y_train)

        # Get the best hyperparameters
        best_params = random_search.best_params_

        # Training an XGBoost Classifier
        model = xgb.XGBClassifier(**best_params)
        model.fit(X_train, y_train)
        joblib.dump(model, 'xgboost_model.pkl')
    else:
        model = joblib.load('xgboost_model.pkl')

    features = extract_features(FOLDER_PATH + audio_file)
    prediction = model.predict(features)
    prediction = np.mean(prediction)

    if prediction < THRESHOLD:
        return 'HUMAN - ' + str(prediction)
    else:
        return 'AI - ' + str(prediction)
