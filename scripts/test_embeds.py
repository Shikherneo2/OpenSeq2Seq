import os
import numpy as np

def return_sum(dir):
	files = [os.path.join(dir, i) for i in os.listdir(dir) if i[-3:]=="npy"]
	embeds = [np.load(i) for i in files]
	embeds = np.array(embeds)
	embeds = np.sum( embeds, axis=-1 )
	return sorted(embeds)

print( return_sum( "/home/sdevgupta/mine/OpenSeq2Seq/logs_mixed_phonemes/logs_highway_net/logs/embeddings" ) )
print( return_sum( "/home/sdevgupta/mine/OpenSeq2Seq/logs_mixed_phonemes/logs_highway_net/logs/embeddings_single_batch" ) )