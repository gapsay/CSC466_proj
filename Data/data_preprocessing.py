import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import train_test_split

# Dataset: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/data

SOURCE_DIR = r"..."  # change this ex C:\Users\chris\Downloads\data_music\genres_original
DEST_DIR = r"...CSC466_proj\Data\processed_data"  # change this
genres = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(
    os.path.join(SOURCE_DIR, d))]

TRAIN_RATIO = 0.8


def make_dirs():
    """Create output directory structure"""
    for split in ["train", "test"]:
        for cls in genres:
            os.makedirs(os.path.join(DEST_DIR, split, cls), exist_ok=True)


def process_audio_file(audio_path, dest_path):
    """Generates a Mel Spectrogram image from a .wav file"""
    try:
        y, sr = librosa.load(audio_path, duration=30)

        # converts to spectogram
        # mel is used here cause it gets rid of high frequencies and basically
        # only keeps frequencies that humans hear which is good for classification
        spectrogram = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
        spectrogram_dB = librosa.power_to_db(spectrogram, ref=np.max)

        # save spectrogram as image
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(spectrogram_dB, sr=sr)
        plt.axis('off')
        plt.savefig(dest_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")


def main():
    make_dirs()
    for genre in genres:
        print(f"Processing genre: {genre}...")

        # path to genre
        genre_path = os.path.join(SOURCE_DIR, genre)

        # loads all files into this list, lowkey should have used a streaming approach but this is easier
        files = [f for f in os.listdir(genre_path) if f.endswith('.wav')]

        train_files, test_files = train_test_split(
            files, test_size=(1 - TRAIN_RATIO), random_state=42)

        for split_name, file_list in [('train', train_files), ('test', test_files)]:
            dest_folder = os.path.join(DEST_DIR, split_name, genre)
            for f in file_list:
                audio_path = os.path.join(genre_path, f)
                dest_path = os.path.join(
                    dest_folder, f.replace('.wav', '.png'))

                # skips image creation if it alr exists
                if not os.path.exists(dest_path):
                    process_audio_file(audio_path, dest_path)


if __name__ == "__main__":
    main()
