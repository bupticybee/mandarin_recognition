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
#import torch
#from torch.utils.data import DataLoader
#from torch.utils.data import Dataset
import tensorflow as tf
import tflearn 
from tflearn.data_flow import DataFlow,DataFlowStatus,FeedDictFlow
import re
import sigprocess
import calcmfcc
from sklearn import preprocessing
import random
import python_speech_features

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

PHON_LABELS = [
 'W',
 'IY',
 'AE',
 'D',
 'EH',
 'DH',
 'NG',
 'T',
 'ZH',
 'UH',
 'K',
 'UW',
 'N',
 'V',
 'AO',
 'AW',
 'TH',
 'CH',
 'OY',
 'AH',
 'L',
 'G',
 'S',
 'AA',
 'Z',
 'M',
 'F',
 'P',
 'EY',
 'Y',
 'B',
 'IH',
 'SH',
 'ER',
 'AY',
 'HH',
 'R',
 'OW',
 'JH',
 ' '
]

label2ind = dict(zip(LABELS,range(1,len(LABELS) + 1)))
ind2label = dict(zip(range(1,len(LABELS) + 1),LABELS))

phlabel2ind = dict(zip(PHON_LABELS,range(1,len(PHON_LABELS) + 1)))
ind2phlabel = dict(zip(range(1,len(PHON_LABELS) + 1),PHON_LABELS))

def get_word2ph_dic(phm_map_file):
    """
    get the word to Phoneme diction based to a tedlium phoneme dict file, retval like 
    'home' => ['HH', 'OW', 'M']
    """
    word2ph = {}
    with open(phm_map_file) as fhdl:
        for line in fhdl:
            linesp = line.strip().split(" ")
            word = linesp[0]
            if 'ERROR' in word:
                continue
            phs = linesp[1:]
            word2ph[word] = phs
    return word2ph

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

def blockedgauss(mu,sigma,nsigma=2):
    while True:
        numb = random.gauss(mu,sigma)
        if (numb > mu - nsigma * sigma and numb < mu + nsigma * sigma):
            break
    return numb

def posrand(truepos=0.8):
    return random.random() < truepos

def resample(smp, scale=1.0):
    """Resample a sound to be a different length

    Sample must be mono.  May take some time for longer sounds
    sampled at 44100 Hz.

    Keyword arguments:
    scale - scale factor for length of sound (2.0 means double length)

    """
    # f*ing cool, numpy can do this with one command
    # calculate new length of sample
    n = round(len(smp) * scale)
    # use linear interpolation
    # endpoint keyword means than linspace doesn't go all the way to 1.0
    # If it did, there are some off-by-one errors
    # e.g. scale=2.0, [1,2,3] should go to [1,1.5,2,2.5,3,3]
    # but with endpoint=True, we get [1,1.4,1.8,2.2,2.6,3]
    # Both are OK, but since resampling will often involve
    # exact ratios (i.e. for 44100 to 22050 or vice versa)
    # using endpoint=False gets less noise in the resampled sound
    return np.interp(
        np.linspace(0.0, 1.0, n, endpoint=False), # where to interpret
        np.linspace(0.0, 1.0, len(smp), endpoint=False), # known positions
        smp, # known data points
        )

def fetch_audio(path,augmentation_speed=False,augmentation_speed_rate=0.12,augmentation_linear=False):
    sample_rate,sound_sc = scipy.io.wavfile.read(path)
    if augmentation_speed == True and posrand(0.8):
        rate = blockedgauss(1,augmentation_speed_rate,1)
        if augmentation_linear == True:
            sound_sc = resample(sound_sc,rate)
        else:
            sound_sc = scipy.signal.resample(sound_sc,num=int(len(sound_sc) * rate))
    return sample_rate,sound_sc * 65536.0 

def fft_feature_extrace_pytorch(sample,n_fft=320,hop_length=160,win_length=320,window=scipy.signal.hamming,normalize=True):
    D = librosa.stft(sample,n_fft=n_fft,hop_length=hop_length,win_length=win_length,window=window)
    spect, phase = librosa.magphase(D)
    # S = log(S+1)
    spect = np.log1p(spect)
    #spect = torch.FloatTensor(spect)
    if normalize:
        mean = np.mean(spect)
        std = np.std(spect)
        spect -= mean
        spect /= std
    return spect


fft_feature_extrace = fft_feature_extrace_pytorch


def get_random_index(weights):
    rand = random.random() * np.sum(weights)
    sumnum = 0
    for i,val in enumerate(weights):
        sumnum += val
        if sumnum > rand:
            return i
    return i
    
def get_rand_noise(noises,weights,length):
    ind = get_random_index(weights)
    onenoise = noises[ind]
    start = random.randint(0,len(onenoise) - length)
    return onenoise[start:start + length]

def augment_VTLP(inputspec,alpha = 1.0,f0 = 0.9,fmax=1):
    # inputspec [freq,time]
    freqbins,timebins = inputspec.shape
    inputspec = np.transpose(inputspec)
    scale = np.linspace(0, 1, freqbins)
    scale = np.array(list(map(lambda x: x * alpha if x <= f0 else (fmax-alpha*f0)/(fmax-f0)*(x-f0)+alpha*f0, scale)))
    scale *= (161-1)/max(scale)
    
    newspec = np.zeros([timebins, freqbins])
    weights = np.zeros(freqbins)
    totw = [0.0 for i in range(freqbins)]
    for i in range(0, freqbins):
        if (i < 1 or i + 1 >= freqbins):
            newspec[:, i] += inputspec[:, i]
            totw[i] += 1.0
            weights[i] += 1
            continue
        else:
            w_up = scale[i] - np.floor(scale[i])
            w_down = 1 - w_up
            j = int(np.floor(scale[i]))

            newspec[:, j] += w_down * inputspec[:, i]
            weights[j] += w_down
            totw[j] += w_down

            newspec[:, j + 1] += w_up * inputspec[:, i]
            weights[j + 1] += w_up
            totw[j + 1] += w_up
    newspec = newspec / weights
    
    outputspec = np.transpose(newspec)
    return outputspec

class SpectrogramDataset(object):
    def __init__(self, manifest_filepath, labels=LABELS, normalize=False,ch2ind=None,ind2ch=None, augment=False,mfcc=False,python_speech_features_imp=False,phon=False,phm_dict_file=None,phon_and_char=False,augmentation_speed=False,augmentation_speed_rate=0.12
                 ,augmentation_linear=False,add_noise=False,noises=None,noise_weight=None,noise_volumn=0.15,random_crop=False,random_crop_rate=0.2,freq_bins=None,augment_vtlp=False,augment_vtlp_alpha=0.1,file_augment_func=None):
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
        self.labels = labels
        self.labels_map = dict([(labels[i], i) for i in range(len(labels))])
        self.mfcc = mfcc
        self.phon = phon
        if phm_dict_file != None:
            self.word2ph = get_word2ph_dic(phm_dict_file)
        self.augmentation_speed = augmentation_speed
        self.augmentation_speed_rate = augmentation_speed_rate
        self.augmentation_linear = augmentation_linear
        self.add_noise = add_noise
        self.noises = noises
        self.noise_weight = noise_weight
        self.random_crop = random_crop
        self.random_crop_rate = random_crop_rate
        self.freq_bins = freq_bins
        self.augment_vtlp = augment_vtlp
        self.augment_vtlp_alpha = augment_vtlp_alpha
        self.phon_and_char = phon_and_char
        self.noise_volumn = noise_volumn
        self.python_speech_features_imp = python_speech_features_imp
        self.file_augment_func = file_augment_func
        self.ch2ind = ch2ind
        self.ind2ch = ind2ch
        #super(SpectrogramDataset, self).__init__(audio_conf, normalize, augment)

    def __getitem__(self, index):
        sample = self.ids[index]
        audio_path, transcript_path = sample[0], sample[1]
        sample_rate,voice = fetch_audio(audio_path)
        spect = fft_feature_extrace_pytorch(voice)
        transcript = self.parse_transcript(transcript_path)
        return spect, transcript

    def parse_transcript(self, transcript_path):
        with open(transcript_path, 'r',encoding='utf-8') as transcript_file:
            transcript = transcript_file.read().replace('\n', '')
            transcript = re.sub(r'\(.*?\)','',transcript)
            transcript = re.sub(r'\{.*?\}','',transcript)
            transcript = re.sub(r'\<.*?\>','',transcript)
        transcript = list(filter(None, [self.labels_map.get(x) for x in list(transcript)]))
        return transcript

    def __len__(self):
        return self.size
    
def _collate_fn(batch):
    def func(p):
        return p[0].shape[1]
    voice_sizes = [p[0].shape[1] for p in batch]
    longest_sample = max(batch, key=func)[0]
    freq_size = longest_sample.shape[0]
    minibatch_size = len(batch)
    max_seqlength = longest_sample.shape[1]
    inputs = np.zeros([minibatch_size, 1, freq_size, max_seqlength])
    target_sizes = np.zeros(minibatch_size)
    targets = []
    for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.shape[1]
        #inputs[x][0].narrow(1, 0, seq_length).copy_(tensor)
        inputs[x,0,:,0:seq_length] = tensor
        target_sizes[x] = len(target)
        targets += target
    #targets = torch.IntTensor(targets)
    #targets = np.asarray(targets)
    return inputs, targets, target_sizes,voice_sizes

 
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
    
def mfcc_feature_extrace(voice,sample_rate):
    feat = calcmfcc.calcfeat_delta_delta(voice,sample_rate)
    feat = preprocessing.scale(feat)
    feat = np.transpose(feat)
    return feat

def parse_transcript_phon(transcript_path,word2ph):
    with open(transcript_path, 'r') as transcript_file:
        transcript = transcript_file.read().replace('\n', '')
        transcript = transcript.split(' ')
        #phlabel2ind
    retval = []
    for word in transcript:
        word = word.lower()
        if word not in word2ph:
            continue
        phs = word2ph[word]
        phs = [phlabel2ind[i] for i in phs]
        retval += phs
        retval.append(phlabel2ind[' '])
    return retval[:-1]
    
class VoicesFlow(SpectrogramDataset):
    def __getitem__(self, indexs):
        batch_item = []
        audio_paths = []
        transcript_paths = []
        for index in indexs:
            sample = self.ids[index]
            audio_path, transcript_path = sample[0], sample[1]
            if self.file_augment_func != None:
                audio_path = self.file_augment_func(audio_path)
            audio_paths.append(audio_path)
            transcript_paths.append(transcript_path)
            sample_rate,voice = fetch_audio(audio_path,self.augmentation_speed,self.augmentation_speed_rate,self.augmentation_linear)
            if self.add_noise and random.random() < 0.5:
                noise = get_rand_noise(self.noises,self.noise_weight,len(voice))
                voice = random.random() * self.noise_volumn * noise + voice
            if self.random_crop == True:
                ts = random.random() * 2 * self.random_crop_rate - self.random_crop_rate
                if ts >= 0:
                    voice = voice[int(ts * len(voice)):]# + 0.1 * noise_sc[:len(sound_sc)]
                elif int(ts * len(voice)) == 0:
                    voice = voice
                else:
                    voice = voice[:int(ts * len(voice))]
            if self.mfcc == False:
                spect = fft_feature_extrace_pytorch(voice)
                #spect = python_speech_features.mfcc(voice / 65536.0).T
            else:
                if self.python_speech_features_imp:
                    spect = python_speech_features.mfcc(voice / 65536.0).T
                else:
                    spect = mfcc_feature_extrace(voice,sample_rate)
            if self.augment_vtlp == True and random.random() > 0.33:
                rand_alpha = 1 + random.random() * self.augment_vtlp_alpha * 2 - self.augment_vtlp_alpha
                spect = augment_VTLP(spect,alpha=rand_alpha)
                
            if self.freq_bins != None:
                spect = spect[:self.freq_bins,:]
            
            if self.phon_and_char == True:
                transcript = [self.parse_transcript(transcript_path),parse_transcript_phon(transcript_path,self.word2ph)]
            else:
                if self.phon == False:
                    transcript = self.parse_transcript(transcript_path)
                else:
                    transcript = parse_transcript_phon(transcript_path,self.word2ph)
            #print("ddd {} aaa {}".format(len(transcript),spect.shape))
            one_item = (spect, transcript)
            batch_item.append(one_item)
        x = _collate_fn(batch_item)
        
        outx = x[0][:,0,:,:]
        outx = np.expand_dims(outx,-1)
        # shape should be [batch_size,freq,time,1]
        arr = x[1]
        indexs = list(x[2][::-1])
        outy = []
        #print(indexs,len(arr))
        while len(indexs) > 0:
            index = indexs.pop()
            index = int(index)
            tmp,arr = arr[:index],arr[index:]
            if self.phon_and_char == True:
                text = ''.join([self.labels[i] for i in tmp[0]])
                outy.append([[self.ch2ind[i] for i in text],tmp[1]])
            else:
                if not self.phon:
                    text = ''.join([self.labels[i] for i in tmp])
                    outy.append([self.ch2ind[i] for i in text])
                    #outy.append(text)
                else:
                    outy.append(tmp)
        #print(len(x))
        return outx,outy,audio_paths,transcript_paths,x[3]

#class AudioDataLoader(DataLoader):
#    def __init__(self, *args, **kwargs):
#        """
#        Creates a data loader for AudioDatasets.
#        """
#        super(AudioDataLoader, self).__init__(*args, **kwargs)
#        self.collate_fn = _collate_fn
#        
def get_flow(data,gpu_core=0,batch_size=16,shuffle=False,threads=8,max_queue=32): 
    with tf.device("/gpu:{}".format(gpu_core)):
        coord = tf.train.Coordinator()
        flow = FeedDictFlow({
                'data':data,
            },coord,batch_size=batch_size,shuffle=shuffle,continuous=True,num_threads=threads,max_queue=max_queue)
        flow.start()
        return flow
    return None
