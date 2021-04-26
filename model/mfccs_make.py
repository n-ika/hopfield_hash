import numpy as np
import librosa
import scipy.io.wavfile as wav
import glob

# folder_train = "./wavs/"
# test_folder = "./test_wavs/"

def make_mfcc(folder):
    """
    Go through the folder and find all (and only) files ending with .wav
    Here, we transform each .wav file into MFCCs and then flatten them into one vector.
    We do this because we want one hash per .wav file.
    
    Any file shorter than the longest file in the folder will be padded with values 0,
    so that all concatenated file vectors are of the same length.
    
    Parameters
    ----------
    folder : path to folder with wav sounds
    
    Returns
    -------
    a list of flattened MFCC vectors
    """
    vectors = []
    for file in glob.glob(folder + "*.wav", recursive=True):
        y, sr = librosa.load(file)
        mfcc_feat = librosa.feature.mfcc(y=y, sr=sr)
        vect = mfcc_feat.flatten()
        vectors.append(vect)
        print(len(vectors), " mfccs done")
    # find the largest vector
    max_length = len(max(vectors, key=lambda p: len(p)))
    # append zeros to all the other vectors
    for i in range(len(vectors)):
        vectors[i] = np.pad(vectors[i], (0,max_length-len(vectors[i])))
    return vectors

