import os
import sys
import tqdm

mel_dir = sys.argv[1]
infer_file = sys.argv[2]
file_prefix = sys.argv[3]

data = open(infer_file, "r").read().split("\n")
lines = [ i.strip().split("|") for i in data if i.strip()!="" ]

for ind,line in tqdm.tqdm(enumerate(lines)):
    filename = line[0]
    id = line[-1]
    # if filename.find("LJSpeech-1.1")!=-1:
    #     location = os.path.basename(filename)
    # else:
    #     location = "_".join(filename.split("/")[-2:])+".npy"
    location = os.path.basename(filename)
    # print( file_prefix+"-"+id+".npy" )
    print( location )
    os.rename( os.path.join(mel_dir, file_prefix+"-"+id+".npy" ), os.path.join(mel_dir, location) )
    # if ind>15:
    #     break