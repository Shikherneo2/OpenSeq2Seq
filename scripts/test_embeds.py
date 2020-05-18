import os
import numpy as np

def return_sum(dir):
	files = [os.path.join(dir, i) for i in os.listdir(dir) if i[-3:]=="npy"]
	files = sorted(files)
	# print([os.path.basename(i) for i in files])
	embeds = [np.load(i) for i in files]
	embeds = np.array(embeds)
	embeds2 = np.sum( embeds, axis=-1 )
	return embeds2

print( return_sum( "/home/sdevgupta/mine/OpenSeq2Seq/logs_mixed_phonemes/logs_highway_net/logs/embeddings_single" ) )
print( return_sum( "/home/sdevgupta/mine/OpenSeq2Seq/logs_mixed_phonemes/logs_highway_net/logs/embeddings_batch_9" ) )