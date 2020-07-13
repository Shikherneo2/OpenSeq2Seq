import os
import sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

indir = sys.argv[1]

hop_length = 256
sampling_rate = 22050
files = [ os.path.join(indir, i) for i in os.listdir(indir) if i[-3:]=="npy"]

buckets = [0 for i in range(9)]
for file in tqdm(files):
	mel_length = float(np.load( file ).shape[1])
	wav_length = int(mel_length*hop_length/sampling_rate)
	if wav_length>8:
		continue
	buckets[wav_length]+=1

plt.scatter(range(9), buckets)
plt.show()
np.save( open("wav_length_frequency.npy", "w"), np.array(buckets))