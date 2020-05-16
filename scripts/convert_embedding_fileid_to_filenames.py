import os
import sys

embeddings_dir = sys.argv[1]
infer_file = sys.argv[2]

data = open(infer_file, "r").read().split("\n")
lines = [ i.strip().split("|") for i in data if i.strip()!="" ]
i = 0
for line in lines:
	filename = line[0]
	id = line[-1]
	location = "_".join(filename.split("/")[-2:])+".npy"
	os.rename( os.path.join(mel_dir, "embed-"+id+".npy" ), os.path.join(mel_dir, location) )
