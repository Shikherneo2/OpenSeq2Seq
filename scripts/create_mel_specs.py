import os
import sys
import librosa
import numpy as np
import librosa.filters

input_dir = sys.argv[1]
output_dir = sys.argv[2]

def get_speech_features_from_file(
		filename,
		num_features,
		n_fft=1024,
		hop_length=None,
		win_length=None,
		mag_power=2,
		feature_normalize=False,
		mean=0.,
		std=1.,
		trim=False,
		data_min=1e-5,
		mel_basis=None,
		sampling_rate_param=16000
):
	""" Helper function to retrieve spectrograms from wav files

	Args:
		  filename (string): WAVE filename.
		  num_features (int): number of speech features in frequency domain.
		  features_type (string): 'magnitude' or 'mel'.
		  n_fft (int): size of analysis window in samples.
		  hop_length (int): stride of analysis window in samples.
		  mag_power (int): power to raise magnitude spectrograms (prior to dot product
			with mel basis)
			1 for energy spectrograms
			2 fot power spectrograms
		  feature_normalize (bool): whether to normalize the data with mean and std
		  mean (float): if normalize is enabled, the mean to normalize to
		  std (float): if normalize is enabled, the deviation to normalize to
		  trim (bool): Whether to trim silence via librosa or not
		  data_min (float): min clip value prior to taking the log.

	Returns:
		  np.array: np.array of audio features with shape=[num_time_steps,
		  num_features].
	"""
	# load audio signal
	if win_length is None:
		win_length = n_fft
	signal, fs = librosa.core.load(filename, sr=sampling_rate_param)
	print("Signal")
	print(signal[800:820])
	if hop_length is None:
		hop_length = int(n_fft / 4)
	if trim:
		signal, _ = librosa.effects.trim(
			signal,
			frame_length=int(n_fft/2),
			hop_length=int(hop_length/2)
		)

	speech_features = get_speech_features(
		signal, fs, num_features, n_fft,
		hop_length, win_length, mag_power, feature_normalize, mean, std, data_min, mel_basis
	)
	return speech_features


def get_speech_features(
		signal,
		fs,
		num_features,
		n_fft=1024,
		hop_length_param=256,
		win_length_param=1024,
		mag_power=2,
		feature_normalize=False,
		mean=0.,
		std=1.,
		data_min=1e-5,
		mel_basis=None
):
	""" Helper function to retrieve spectrograms from loaded wav

	Args:
		  signal: signal loaded with librosa.
		  fs (int): sampling frequency in Hz.
		  num_features (int): number of speech features in frequency domain.
		  features_type (string): 'magnitude' or 'mel'.
		  n_fft (int): size of analysis window in samples.
		  hop_length (int): stride of analysis window in samples.
		  mag_power (int): power to raise magnitude spectrograms (prior to dot product
			with mel basis)
			1 for energy spectrograms
			2 fot power spectrograms
		  feature_normalize(bool): whether to normalize the data with mean and std
		  mean(float): if normalize is enabled, the mean to normalize to
		  std(float): if normalize is enabled, the deviation to normalize to
		  data_min (float): min clip value prior to taking the log.

	Returns:
		  np.array: np.array of audio features with shape=[num_time_steps,
		  num_features].
	"""
	num_features_mel = num_features

	complex_spec = librosa.stft( y=signal, n_fft=n_fft, hop_length=hop_length_param, win_length=win_length_param )
	mag, _ = librosa.magphase( complex_spec, power=mag_power )

	print("Mag")
	print(mag.shape)
	print(mag[300][200:220])
	if mel_basis is None:
		htk = True
		norm = None
		
		mel_basis = librosa.filters.mel(
			sr=fs,
			n_fft=n_fft,
			n_mels=num_features_mel,
			htk=htk,
			norm=norm
		)
		
	print("mel_basis")
	print(mel_basis.shape)
	print(np.mean(mel_basis))
	features = np.dot(mel_basis, mag)
	print("features before log and clamp")
	print(features[20][250:260])
	features = np.log(np.clip(features, a_min=data_min, a_max=8000))
	print("Final features before transpose")
	print(features[20][250:260])
	return features.T

for file in os.listdir(input_dir ):
	if( file.split(".")[-1] =="wav" ):
		mel = get_speech_features_from_file(  os.path.join(input_dir, file),
										num_features=80,
										n_fft=800,
										hop_length=200,
										win_length=800,
										mag_power=2,
										feature_normalize=False,
										mean=0.,
										std=1.,
										trim=False,
										data_min=0,
										mel_basis=None,
										sampling_rate_param=16000 )
		print( mel.shape )
		# np.save( os.path.join(output_dir, file.replace("wav","npy")), mel )
