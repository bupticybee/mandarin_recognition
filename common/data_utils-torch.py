import os
import argparse
import subprocess
import unicodedata
from tqdm import tqdm
import io
import os
import subprocess
from utils import ProgressBar
import fnmatch
import librosa
import scipy
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import tensorflow as tf
import tflearn 
from tflearn.data_flow import DataFlow,DataFlowStatus,FeedDictFlow

TED_LIUM_V2_DL_URL = "http://www.openslr.org/resources/19/TEDLIUM_release2.tar.gz"
LABELS = [
  "_",
  "'",
  "A",
  "B",
  "C",
  "D",
  "E",
  "F",
  "G",
  "H",
  "I",
  "J",
  "K",
  "L",
  "M",
  "N",
  "O",
  "P",
  "Q",
  "R",
  "S",
  "T",
  "U",
  "V",
  "W",
  "X",
  "Y",
  "Z",
  " "
]

def create_manifest(data_path, tag, ordered=True):
    manifest_path = '%s_manifest.csv' % tag
    file_paths = []
    wav_files = [os.path.join(dirpath, f)
                 for dirpath, dirnames, files in os.walk(data_path)
                 for f in fnmatch.filter(files, '*.wav')]
    for file_path in tqdm(wav_files, total=len(wav_files)):
        file_paths.append(file_path.strip())
    print('\n')
    if ordered:
        _order_files(file_paths)
    with io.FileIO(manifest_path, "w") as file:
        for wav_path in tqdm(file_paths, total=len(file_paths)):
            transcript_path = wav_path.replace('/wav/', '/txt/').replace('.wav', '.txt')
            sample = os.path.abspath(wav_path) + ',' + os.path.abspath(transcript_path) + '\n'
            file.write(sample.encode('utf-8'))
    print('\n')


def _order_files(file_paths):
    print("Sorting files by length...")

    def func(element):
        output = subprocess.check_output(
            ['soxi -D \"%s\"' % element.strip()],
            shell=True
        )
        return float(output)

    file_paths.sort(key=func)

def get_utterances_from_stm(stm_file):
    """
    Return list of entries containing phrase and its start/end timings
    :param stm_file:
    :return:
    """
    res = []
    with io.open(stm_file, "r", encoding='utf-8') as f:
        for stm_line in f:
            tokens = stm_line.split()
            start_time = float(tokens[3])
            end_time = float(tokens[4])
            filename = tokens[0]
            transcript = unicodedata.normalize("NFKD",
                                               " ".join(t for t in tokens[6:]).strip()). \
                encode("utf-8", "ignore").decode("utf-8", "ignore")
            if transcript != "ignore_time_segment_in_scoring":
                res.append({
                    "start_time": start_time, "end_time": end_time,
                    "filename": filename, "transcript": transcript
                })
        return res


def cut_utterance(src_sph_file, target_wav_file, start_time, end_time, sample_rate=16000):
    subprocess.call(["sox {}  -r {} -b 16 -c 1 {} trim {} ={}".format(src_sph_file, str(sample_rate),
                                                                      target_wav_file, start_time, end_time)],
                    shell=True)


def _preprocess_transcript(phrase):
    return phrase.strip().upper()


def filter_short_utterances(utterance_info, min_len_sec=1.0):
    return utterance_info["end_time"] - utterance_info["start_time"] > min_len_sec

sample_rate=16000
def prepare_dir(ted_dir):
    converted_dir = os.path.join(ted_dir, "converted")
    # directories to store converted wav files and their transcriptions
    wav_dir = os.path.join(converted_dir, "wav")
    if not os.path.exists(wav_dir):
        os.makedirs(wav_dir)
    txt_dir = os.path.join(converted_dir, "txt")
    if not os.path.exists(txt_dir):
        os.makedirs(txt_dir)
    counter = 0
    entries = os.listdir(os.path.join(ted_dir, "sph"))
    for sph_file in tqdm(entries, total=len(entries)):
        speaker_name = sph_file.split('.sph')[0]

        sph_file_full = os.path.join(ted_dir, "sph", sph_file)
        stm_file_full = os.path.join(ted_dir, "stm", "{}.stm".format(speaker_name))

        assert os.path.exists(sph_file_full) and os.path.exists(stm_file_full)
        all_utterances = get_utterances_from_stm(stm_file_full)

        all_utterances = filter(filter_short_utterances, all_utterances)
        for utterance_id, utterance in enumerate(all_utterances):
            target_wav_file = os.path.join(wav_dir, "{}_{}.wav".format(utterance["filename"], str(utterance_id)))
            target_txt_file = os.path.join(txt_dir, "{}_{}.txt".format(utterance["filename"], str(utterance_id)))
            cut_utterance(sph_file_full, target_wav_file, utterance["start_time"], utterance["end_time"],
                          sample_rate=sample_rate)
            with io.FileIO(target_txt_file, "w") as f:
                f.write(_preprocess_transcript(utterance["transcript"]).encode('utf-8'))
        counter += 1

def fetch_audio(path):
    sample_rate,sound_sc = scipy.io.wavfile.read(path)
    return sample_rate,sound_sc * 65536.0 

def fft_feature_extrace_pytorch(sample,n_fft=320,hop_length=160,win_length=320,window=scipy.signal.hamming,normalize=True):
    D = librosa.stft(sample,n_fft=n_fft,hop_length=hop_length,win_length=win_length,window=window)
    spect, phase = librosa.magphase(D)
    # S = log(S+1)
    spect = np.log1p(spect)
    spect = torch.FloatTensor(spect)
    if normalize:
        mean = spect.mean()
        std = spect.std()
        spect.add_(-mean)
        spect.div_(std)
    return spect

def fft_feature_extrace_numpy(sample,n_fft=320,hop_length=160,win_length=320,window=scipy.signal.hamming,normalize=True):
    D = librosa.stft(sample,n_fft=n_fft,hop_length=hop_length,win_length=win_length,window=window)
    spect, phase = librosa.magphase(D)
    # S = log(S+1)
    spect = np.log1p(spect)
    spect = torch.FloatTensor(spect)
    if normalize:
        mean = spect.mean()
        std = spect.std()
        spect = spect - mean
        spect = spect / std
    return spect

fft_feature_extrace = fft_feature_extrace_pytorch



class SpectrogramDataset(Dataset):
    def __init__(self, manifest_filepath, labels=LABELS, normalize=False, augment=False):
        """
        Dataset that loads tensors via a csv containing file paths to audio files and transcripts separated by
        a comma. Each new line is a different sample. Example below:

        /path/to/audio.wav,/path/to/audio.txt
        ...

        :param audio_conf: Dictionary containing the sample rate, window and the window length/stride in seconds
        :param manifest_filepath: Path to manifest csv as describe above
        :param labels: String containing all the possible characters to map to
        :param normalize: Apply standard mean and deviation normalization to audio tensor
        :param augment(default False):  Apply random tempo and gain perturbations
        """
        with open(manifest_filepath) as f:
            ids = f.readlines()
        ids = [x.strip().split(',') for x in ids]
        self.ids = ids
        self.size = len(ids)
        self.labels_map = dict([(labels[i], i) for i in range(len(labels))])
        #super(SpectrogramDataset, self).__init__(audio_conf, normalize, augment)

    def __getitem__(self, index):
        sample = self.ids[index]
        audio_path, transcript_path = sample[0], sample[1]
        sample_rate,voice = fetch_audio(audio_path)
        spect = fft_feature_extrace_pytorch(voice)
        transcript = self.parse_transcript(transcript_path)
        return spect, transcript

    def parse_transcript(self, transcript_path):
        with open(transcript_path, 'r') as transcript_file:
            transcript = transcript_file.read().replace('\n', '')
        transcript = list(filter(None, [self.labels_map.get(x) for x in list(transcript)]))
        return transcript

    def __len__(self):
        return self.size
    
def _collate_fn(batch):
    def func(p):
        return p[0].size(1)

    longest_sample = max(batch, key=func)[0]
    freq_size = longest_sample.size(0)
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(1)
    inputs = torch.zeros(minibatch_size, 1, freq_size, max_seqlength)
    input_percentages = torch.FloatTensor(minibatch_size)
    target_sizes = torch.IntTensor(minibatch_size)
    targets = []
    for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(1)
        inputs[x][0].narrow(1, 0, seq_length).copy_(tensor)
        input_percentages[x] = seq_length / float(max_seqlength)
        target_sizes[x] = len(target)
        targets.extend(target)
    targets = torch.IntTensor(targets)
    return inputs, targets, input_percentages, target_sizes

 
class SpectrogramFlow(SpectrogramDataset):
    def __getitem__(self, indexs):
        batch_item = []
        for index in indexs:
            sample = self.ids[index]
            audio_path, transcript_path = sample[0], sample[1]
            sample_rate,voice = fetch_audio(audio_path)
            spect = fft_feature_extrace_pytorch(voice)
            transcript = self.parse_transcript(transcript_path)
            one_item = (spect, transcript)
            batch_item.append(one_item)
        return _collate_fn(batch_item)



class AudioDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn
        
def get_flow(data,gpu_core=0,batch_size=16,shuffle=False): 
    with tf.device("/gpu:{}".format(gpu_core)):
        coord = tf.train.Coordinator()
        flow = FeedDictFlow({
                'data':data,
            },coord,batch_size=batch_size,shuffle=shuffle)
        flow.start()
        return flow
    return None
