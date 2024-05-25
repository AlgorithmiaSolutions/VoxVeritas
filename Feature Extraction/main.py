import librosa
import pandas as pd
import numpy as np


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
        'chroma': chroma,
        'rms': rms,
        'spectral_centroid': spectral_centroid,
        'spectral_bandwidth': spectral_bandwidth,
        'spectral_rolloff': spectral_rolloff,
        'zero_crossing_rate': zero_crossing_rate
    }

    # Include MFCCs separately
    for i in range(20):
        features[f'mfcc_{i + 1}'] = mfccs[i, :]

    # Compute mean, median and standard deviation
    aggregated_features = {key: [np.mean(value), np.median(value), np.std(value)] for key, value in features.items()}
    return aggregated_features


def extract_features(file_path, label):
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

        # Separate mean, median and standard deviation into columns
        features = {}
        for key, values in segment_features.items():
            features[f'{key}_mean'] = values[0]
            features[f'{key}_median'] = values[1]
            features[f'{key}_std'] = values[2]

        # Add label column (AI = 1; Human = 0)
        features['label'] = label

        df = df._append(features, ignore_index=True)

    return df


# Extracting features from the human voice audio file
df_features = extract_features('human.mp3', 0)

# Extracting features from the AI voice audio file and appending it to dataframe
df_features = df_features._append(extract_features('human.mp3', 1))

# Saving it to a csv file
df_features.to_csv('output.csv', index=False)
