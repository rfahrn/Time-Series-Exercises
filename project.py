## This script is a part of the project "Sleep Stage Classification using EEG Signals" for the course "AI for Medical Time Series" at the University of Berne.
import sys
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import mne
import matplotlib
import matplotlib.pyplot as plt
from mne.datasets.sleep_physionet.age import fetch_data
import pywt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
from sklearn.model_selection import cross_val_score, permutation_test_score
import torch.optim as optim
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset, random_split


class DataHandler:
    def __init__(self, subjects, recordings, load_eeg_only=True, crop_wake_mins=30):
        self.subjects = subjects
        self.recordings = recordings
        self.load_eeg_only = load_eeg_only
        self.crop_wake_mins = crop_wake_mins
        self.fnames = fetch_data(subjects=self.subjects, recording=self.recordings, on_missing='warn')

    def loading_raw(self, raw_fname, annot_fname):
        mapping = {
            'EOG horizontal': 'eog',
            'Resp oro-nasal': 'misc',
            'EMG submental': 'misc',
            'Temp rectal': 'misc',
            'Event marker': 'misc'
        }
        exclude = mapping.keys() if self.load_eeg_only else ()
        raw = mne.io.read_raw_edf(raw_fname, exclude=exclude, infer_types=True, preload=True, stim_channel="Event marker", verbose="error")
        annots = mne.read_annotations(annot_fname)
        raw.set_annotations(annots, emit_warning=False)
        if not self.load_eeg_only:
            raw.set_channel_types(mapping)
        if self.crop_wake_mins > 0:
            mask = [x[-1] in ['1', '2', '3', '4', 'R'] for x in annots.description]
            sleep_event_inds = np.where(mask)[0]
            tmin = max(0, annots[int(sleep_event_inds[0])]['onset'] - self.crop_wake_mins * 60)
            tmax = min(raw.times[-1], annots[int(sleep_event_inds[-1])]['onset'] + self.crop_wake_mins * 60)
            raw.crop(tmin=tmin, tmax=tmax)
        ch_names = {i: i.replace('EEG ', '') for i in raw.ch_names if 'EEG' in i}
        mne.rename_channels(raw.info, ch_names)
        basename = os.path.basename(raw_fname)
        subj_nb, rec_nb = int(basename[3:5]), int(basename[5])
        raw.info['subject_info'] = {'id': subj_nb, 'rec_id': rec_nb}
        return raw

    def fetch_demographics(self):
        participant_ages, participant_gender = [], []
        for file_info in self.fnames:
            raw = mne.io.read_raw_edf(file_info[0], preload=True)
            subject_info = raw.info.get('subject_info', {})
            if 'last_name' in subject_info:
                age_str = subject_info['last_name']
                age = int(age_str.rstrip('yr'))
                participant_ages.append(age)
            if 'sex' in subject_info:
                gender = 'male' if subject_info['sex'] == 1 else 'female'
                participant_gender.append(gender)
        return participant_ages, participant_gender

    def create_subset(self, ages, min_age=30, max_age=80):
        valid_subjects = [i for i, age in enumerate(ages) if min_age <= age <= max_age]
        return [self.fnames[i] for i in valid_subjects]

    def extract_epochs(self, raw_data):
        events, event_id = mne.events_from_annotations(raw_data, event_id={
            'Sleep stage W': 1,
            'Sleep stage 1': 2,
            'Sleep stage 2': 3,
            'Sleep stage 3': 4,
            'Sleep stage 4': 4,
            'Sleep stage R': 5
        })
        tmax = 30.0 - 1.0 / raw_data.info['sfreq']
        epochs = mne.Epochs(raw_data, events, event_id, tmin=0., tmax=tmax, baseline=None, preload=True)
        return epochs


class Visualizer:
    @staticmethod
    def plot_age_distribution(ages):
        plt.figure(figsize=(10, 6))
        plt.hist(ages, bins=10, edgecolor='black', alpha=0.7)
        plt.xlabel("Age")
        plt.ylabel("Number of Subjects")
        plt.title("Age Distribution")
        plt.show()

    @staticmethod
    def plot_raw(raw):
        raw.plot(duration=60, n_channels=30, scalings={'eeg': 75e-6})
    
    @staticmethod
    def plot_psd(raw):
        raw.plot_psd()

    @staticmethod
    def plot_average_by_stage(epochs_list, title):
        stages = range(1, 6)
        colors = ['red', 'green', 'blue', 'orange', 'purple']
        plt.figure(figsize=(15, 5))
        for stage, color in zip(stages, colors):
            stage_data = np.concatenate([epochs.get_data()[epochs.events[:, 2] == stage] for epochs in epochs_list], axis=0)
            average_stage_data = np.mean(stage_data, axis=0)
            plt.plot(average_stage_data.mean(axis=0), label=f'Stage {stage}', color=color)
        plt.title(title)
        plt.xlabel('Time (samples)')
        plt.ylabel('EEG Amplitude')
        plt.legend()
        plt.show()

    @staticmethod
    def plot_feature_importance(importances, feature_names):
        sorted_idx = importances.argsort()
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(importances)), importances[sorted_idx], align='center')
        plt.yticks(range(len(importances)), np.array(feature_names)[sorted_idx])
        plt.xlabel('Importance')
        plt.title('Feature Importance')
        plt.show()

    @staticmethod
    def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
        

class FeatureExtractor:
    @staticmethod
    def calculate_band_power(epochs):
        bands = {
            'Delta': (1, 4),
            'Theta': (4, 8),
            'Alpha': (8, 12),
            'Beta': (12, 30),
            'Gamma': (30, 40)
        }
        features = {}
        plt.figure(figsize=(10, 6))
        for band, (l_freq, h_freq) in sorted(bands.items()):
            epochs_band = epochs.copy().filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin')
            data = epochs_band.get_data()
            band_power = np.log10(np.mean(data**2, axis=2))
            features[band] = band_power.mean(axis=0)
            plt.hist(band_power.ravel(), bins=40, alpha=0.7, label=f'{band} band')
        plt.title('Band Power Distribution Across Different EEG Bands')
        plt.xlabel('Log Power')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()
        return features

    @staticmethod
    def perform_pca(epochs):
        data = epochs.get_data().reshape(len(epochs), -1)
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(data)
        plt.figure(figsize=(8, 6))
        plt.scatter(pca_data[:, 0], pca_data[:, 1], c=np.linspace(0, 1, len(pca_data)), cmap='viridis', alpha=0.5)
        plt.colorbar(label='Index of Epoch')
        plt.title('PCA of EEG Data')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.grid(True)
        plt.show()
        print("Explained variance ratio:", pca.explained_variance_ratio_)


    @staticmethod
    def get_dwt_features(epochs, wavelet='db4', level=5):
        data = epochs.get_data()  # (n_epochs, n_channels, n_times)
        coefficients = []
        for epoch_data in data:
            channel_coeffs = []
            for channel_data in epoch_data:
                coeffs = pywt.wavedec(channel_data, wavelet, level=level)
                concatenated_coeffs = np.hstack([c.ravel() for c in coeffs])
                channel_coeffs.append(concatenated_coeffs)
            coefficients.append(np.mean(channel_coeffs, axis=0))
        return np.array(coefficients)

    @staticmethod
    def plot_dwt_features(group1, group2, title1, title2):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        for ax, group, title in zip(axes, [group1, group2], [title1, title2]):
            concatenated_data = mne.concatenate_epochs(group)
            features = FeatureExtractor.get_dwt_features(concatenated_data)
            ax.imshow(np.log1p(np.abs(features)), aspect='auto', cmap='viridis', origin='lower')
            ax.set_title(title)
            ax.set_xlabel('DWT Coefficient Index')
            ax.set_ylabel('Epoch Index')
        plt.colorbar(axes[1].images[0], ax=axes, orientation='vertical', fraction=.1)
        plt.tight_layout()
        plt.show()


class SleepClassifier(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SleepClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Classifier:
    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.001, batch_size=32, epochs=20):
        self.model = SleepClassifier(input_dim, hidden_dim, output_dim)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.batch_size = batch_size
        self.epochs = epochs

    def train(self, train_loader):
        self.model.train()
        for epoch in range(self.epochs):
            running_loss = 0.0
            for inputs, labels in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')

    def evaluate(self, test_loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'Accuracy: {100 * correct / total}%')

    def logistic_regression(self, features_list, labels):
        all_features = np.vstack([np.hstack([features[band] for band in sorted(features)]) for features in features_list])
        X_train, X_test, y_train, y_test = train_test_split(all_features, labels, test_size=0.3, random_state=42)
        model = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        print(classification_report(y_test, predictions))
        cm = confusion_matrix(y_test, predictions)
        Visualizer.plot_confusion_matrix(cm, classes=['Under 30', 'Over 80', 'Others'])

    def svm_classification(self, features_list, labels):
        all_features = np.vstack([np.hstack([features[band] for band in sorted(features)]) for features in features_list])
        X_train, X_test, y_train, y_test = train_test_split(all_features, labels, test_size=0.3, random_state=42)
        model = SVC(kernel='rbf', C=1.0, decision_function_shape='ovo')
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        print(classification_report(y_test, predictions))
        cm = confusion_matrix(y_test, predictions)
        Visualizer.plot_confusion_matrix(cm, classes=['Under 30', 'Over 80', 'Others'])

    def run_permutation_test(self, grouped_epochs, cls):
        """ Run permutation test to evaluate the significance of the model over time."""
        labels = []
        total_num_time_points = 0
        for epochs, age in grouped_epochs:
            if epochs is not None:
                if age <= 30 or age >= 80:
                    if age <= 30:
                        label = 0
                    elif age >= 80:
                        label = 1
                    labels.append(label)
                    if epochs.get_data().shape[2] > total_num_time_points:
                        total_num_time_points = epochs.get_data().shape[2]
        num_feature_values = [1, 2, 3, 4, 5, *range(10, total_num_time_points, 10)]
        scores = []
        all_permutation_scores = []
        p_values = []
        for num_features in num_feature_values:
            print(f'training with {num_features} time points')
            features_list = []
            for epochs, age in grouped_epochs:
                if epochs is not None:
                    if age <= 30 or age >= 80:
                        features_list.append(FeatureExtractor.calculate_band_power(epochs))
            features = np.vstack([np.hstack([features[band][:num_features] for band in sorted(features)]) for features in features_list])
            score, permutation_scores, p_value = permutation_test_score(cls(), features, labels, cv=5, scoring='f1_macro')
            scores.append(score)
            all_permutation_scores.append(permutation_scores)
            p_values.append(p_value)
        plt.figure(figsize=(8, 6))
        plt.plot(num_feature_values, p_values)
        plt.axhline(0.05, ls="--", color="r")
        plt.title('Test Probability over Time')
        plt.ylabel('p-value')
        plt.xlabel('number of time points')
        plt.show()
        plt.figure(figsize=(8, 6))
        plt.hist(all_permutation_scores[-1], bins=20)
        plt.axvline(scores[-1], ls="--", color="r")
        score_label = f"Score on original\ndata: {scores[-1]:.2f}\n(p-value: {p_values[-1]:.3f})"
        plt.text(0.7, 10, score_label, fontsize=12)
        plt.title('Permutation Scores for Model with all Time Points')
        plt.ylabel('frequency')
        plt.xlabel('F1 score (macro averaged)')
        plt.show()


def main(action, **kwargs):
    """
    Executes the specified action based on the given parameters.

    Parameters:
    - action (str): The action to be executed.
    - kwargs (dict): Additional keyword arguments for specific actions.
    """

    subjects = [i for i in range(83) if i not in [36, 39, 68, 52, 69, 78, 79, 13]]
    recordings = [1]
    data_handler = DataHandler(subjects, recordings)
    ages, genders = data_handler.fetch_demographics()
    visualizer = Visualizer()

    if action == 'plot_age_distribution':
        visualizer.plot_age_distribution(ages)
    elif action == 'plot_raw':
        subject_id = kwargs.get('subject_id', 0)
        raw = data_handler.loading_raw(data_handler.fnames[subject_id][0], data_handler.fnames[subject_id][1])
        visualizer.plot_raw(raw)
    elif action == 'plot_psd':
        subject_id = kwargs.get('subject_id', 0)
        raw = data_handler.loading_raw(data_handler.fnames[subject_id][0], data_handler.fnames[subject_id][1])
        visualizer.plot_psd(raw)
    elif action == 'plot_average_by_stage':
        min_age = kwargs.get('min_age', 30)
        max_age = kwargs.get('max_age', 80)
        subset_fnames = data_handler.create_subset(ages, min_age=min_age, max_age=max_age)
        raws = [data_handler.loading_raw(f[0], f[1]) for f in subset_fnames]
        epochs_list = [data_handler.extract_epochs(raw) for raw in raws]
        visualizer.plot_average_by_stage(epochs_list, f'Average EEG Data by Sleep Stage for Subjects {min_age}-{max_age}')
    elif action == 'train_and_evaluate':
        subset_fnames = data_handler.create_subset(ages)
        raws = [data_handler.loading_raw(f[0], f[1]) for f in subset_fnames]
        epochs_list = [data_handler.extract_epochs(raw) for raw in raws]
        inputs = np.concatenate([epochs.get_data().reshape(len(epochs), -1) for epochs in epochs_list])
        labels = np.concatenate([epochs.events[:, -1] for epochs in epochs_list])
        inputs = torch.tensor(inputs, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
        dataset = TensorDataset(inputs, labels)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        input_dim = inputs.shape[1]
        classifier = Classifier(input_dim=input_dim, hidden_dim=128, output_dim=5)
        classifier.train(train_loader)
        classifier.evaluate(test_loader)
    else:
        print(f"Unknown action: {action}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sleep data analysis")
    parser.add_argument("action", type=str, help="Action to perform: plot_age_distribution, plot_raw, plot_psd, plot_average_by_stage, train_and_evaluate")
    parser.add_argument("--subject_id", type=int, default=0, help="Subject ID for the analysis")
    parser.add_argument("--min_age", type=int, default=30, help="Minimum age for filtering")
    parser.add_argument("--max_age", type=int, default=80, help="Maximum age for filtering")
    args = parser.parse_args()
    main(args.action, subject_id=args.subject_id, min_age=args.min_age, max_age=args.max_age)