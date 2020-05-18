FORMAT FOR FILE LISTS

train, val
wav_filename(relative to dataset location), raw_transcript, transcript

infer
wav_filename(full filepaths), raw_transcript, fileid

params
save_embeddings -- Save style embedding as npy wavs named on fileids
use_npy_wavs -- Load wav files from npys.
use_phonemes -- Use a mixed phoneme and char input model.

scripts -- 
call convert_train_lists.py to convert file lists with npy files from lambda/local machine to file lists with wav files.
call create_embeddings_infer_filelist.py to take in a file list and create a file with fileids at the end. This can be used in infer mode.

You can then call create_dataset_file.py in Text2Style to convert this file into a file that can be used by the model.