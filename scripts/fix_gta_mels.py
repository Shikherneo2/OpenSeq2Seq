import numpy as np
import os
import tqdm
import sys

dataset = open("/home/sdevgupta/mine/Text2Style/open_seq2seq/dataset/combined_dataset_for_text2style_training.csv").read().split("\n")

mel_dir = sys.argv[1]
outdir = sys.argv[2]

for data in tqdm.tqdm( dataset ):
    tokens = data.split("|")
    filepath = tokens[0].strip()
    filename = os.path.basename( tokens[0].strip() )
    fileid = int(tokens[-1].strip())
    #if filename.find("LJ")==-1:
    #    ground_truth = np.load(os.path.join(indir_cath, filename))
    #else:
    #    ground_truth = np.load(os.path.join(indir_lj, filename))
    ground_truth = np.load( filepath ).T
    mel = np.load( os.path.join(mel_dir, "mel-"+str(fileid)+".npy") )
    np.save( os.path.join(outdir, filename), mel[:ground_truth.shape[0],:].T  )
