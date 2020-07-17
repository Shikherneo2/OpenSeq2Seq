import os
import sys
import tqdm

mel_dir = sys.argv[1]
infer_file = sys.argv[2]
file_prefix = sys.argv[3]

data = open(infer_file, "r").read().split("\n")
lines = [ i.strip().split("|") for i in data if i.strip()!="" ]

for line in tqdm.tqdm(lines):
	filename = line[0]
	id = line[-1]
	location = "_".join(filename.split("/")[-2:])+".npy"
	os.rename( os.path.join(mel_dir, file_prefix+"-"+id+".npy" ), os.path.join(mel_dir, location) )