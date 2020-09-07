# pylint: skip-file
import os
import argparse
import tensorflow as tf
from open_seq2seq.losses import Text2SpeechLoss
from open_seq2seq.encoders import Tacotron2Encoder
from open_seq2seq.decoders import Tacotron2Decoder
from open_seq2seq.data import Text2SpeechDataLayer
from open_seq2seq.models import Text2SpeechTacotron
from open_seq2seq.optimizers.lr_policies import fixed_lr, transformer_policy, exp_decay

base_model = Text2SpeechTacotron

output_type = "mel"
base_location = "/home/sdevgupta/mine/OpenSeq2Seq"
dataset_location = os.path.join( base_location, "dataset/" )
# logdir_location = os.path.join( base_location, "logs_mixed_phonemes/logs_highway_net/logs")
logdir_location = os.path.join( base_location, "/home/sdevgupta/mine/OpenSeq2Seq/ljspeech_catheryn_logs" )

# Use npy of wavs instead of loading the wavs and decoding them
use_npy_wavs = True
# Use phonemes instead of raw characters
use_phonemes = True
# Instead of calculating the embedding from the reference wavs, use these saved embeddings
use_saved_embedding = False

# saved_embedding_location = os.path.join( base_location, "logs_mixed_phonemes/logs_highway_net/logs/val_text2_style_dataset_60K_single_batch"  )
saved_embedding_location = "/home/sdevgupta/mine/OpenSeq2Seq/ljspeech_catheryn_logs/embeddings"

# Sound features
trim = False
mag_num_feats = 513
train = "train_cleaned_lambda.csv"
val = "val_cleaned.csv"
# infer = os.path.join( dataset_location, "test_embedding_creation.csv")
#infer = "/home/sdevgupta/mine/Text2Style/open_seq2seq/dataset/combined_dataset_for_text2style_training.csv"
infer = "/home/sdevgupta/mine/data/cath_gta_rand_filelist_gta.txt"

exp_mag = False
if output_type == "magnitude":
  num_audio_features = mag_num_feats
  data_min = 1e-5
elif output_type == "mel":
  num_audio_features = 80
  data_min = 1e-2
elif output_type == "both":
  num_audio_features = {
      "mel": 80,
      "magnitude": mag_num_feats
  }
  data_min = {
      "mel": 1e-2,
      "magnitude": 1e-5,
  }
  exp_mag = True
else:
  raise ValueError("Unknown param for output_type")

base_params = {
  "verbose_inference": False,
	"save_mels": True,
  "save_embeddings": False,
	"gta_force_inference": True,
  
  "random_seed": 0,
  "use_horovod": False,
  "num_gpus": 1,
  "num_epochs": 1000,

  "batch_size_per_gpu": 1,

  "save_summaries_steps": 50,
  "print_loss_steps": 50,
  "print_samples_steps": 1000,
  "eval_steps": 1000,
  "save_checkpoint_steps": 3000,
  "save_to_tensorboard": True,
  "logdir": logdir_location,
  "max_grad_norm":1.,

  "optimizer": "Adam",
  "optimizer_params": {},
  "lr_policy": exp_decay,
  "lr_policy_params": {
    "learning_rate": 1e-3,
    "decay_steps": 15000,
    "decay_rate": 0.1,
    "use_staircase_decay": False,
    "begin_decay_at": 55000,
    "min_lr": 1e-5,
  },
  "dtype": tf.float32,
  "regularizer": tf.contrib.layers.l2_regularizer,
  "regularizer_params": {
    'scale': 1e-6
  },
  "initializer": tf.contrib.layers.xavier_initializer,

  "summaries": ['learning_rate', 'variables', 'gradients', 'gradient_norm', 'global_gradient_norm', 'variable_norm'],

  "encoder": Tacotron2Encoder,
  "encoder_params": {
    "save_embeddings": False,
    "use_saved_embedding": use_saved_embedding,

    "cnn_dropout_prob": 0.5,
    "rnn_dropout_prob": 0.,
    'src_emb_size': 512,
    "conv_layers": [
      {
        "kernel_size": [5], "stride": [1],
        "num_channels": 512, "padding": "SAME"
      },
      {
        "kernel_size": [5], "stride": [1],
        "num_channels": 512, "padding": "SAME"
      },
      {
        "kernel_size": [5], "stride": [1],
        "num_channels": 512, "padding": "SAME"
      }
    ],
    "activation_fn": tf.nn.relu,

    "num_rnn_layers": 1,
    "rnn_cell_dim": 256,
    "rnn_unidirectional": False,
    "use_cudnn_rnn": True,
    "rnn_type": tf.contrib.cudnn_rnn.CudnnLSTM,
    "zoneout_prob": 0.,

    "data_format": "channels_last",

    "style_embedding_enable": True,
    "style_embedding_params": {
      "conv_layers": [
        {
          "kernel_size": [3,3], "stride": [2,2],
          "num_channels": 32, "padding": "SAME"
        },
        {
          "kernel_size": [3,3], "stride": [2,2],
          "num_channels": 32, "padding": "SAME"
        },
        {
          "kernel_size": [3,3], "stride": [2,2],
          "num_channels": 64, "padding": "SAME"
        },
        {
          "kernel_size": [3,3], "stride": [2,2],
          "num_channels": 64, "padding": "SAME"
        },
        {
          "kernel_size": [3,3], "stride": [2,2],
          "num_channels": 128, "padding": "SAME"
        },
        {
          "kernel_size": [3,3], "stride": [2,2],
          "num_channels": 128, "padding": "SAME"
        }
      ],
      "num_rnn_layers": 1,
      "rnn_cell_dim": 128,
      "rnn_unidirectional": True,
      "rnn_type": tf.nn.rnn_cell.GRUCell,
      "emb_size": 512,
      'attention_layer_size': 512,
      "num_tokens": 32,
      "num_heads": 8
    }
  },

  "decoder": Tacotron2Decoder,
  "decoder_params": {
    "zoneout_prob": 0.,
    "dropout_prob": 0.1,
    
    'attention_type': 'location',
    'attention_layer_size': 128,
    'attention_bias': True,

    'decoder_cell_units': 1024,
    'decoder_cell_type': tf.nn.rnn_cell.LSTMCell,
    'decoder_layers': 2,
    
    'enable_prenet': True,
    'prenet_layers': 2,
    'prenet_units': 256,

    'enable_postnet': True,
    "postnet_keep_dropout_prob": 0.5,
    "postnet_data_format": "channels_last",
    "postnet_conv_layers": [
      {
        "kernel_size": [5], "stride": [1],
        "num_channels": 512, "padding": "SAME",
        "activation_fn": tf.nn.tanh
      },
      {
        "kernel_size": [5], "stride": [1],
        "num_channels": 512, "padding": "SAME",
        "activation_fn": tf.nn.tanh
      },
      {
        "kernel_size": [5], "stride": [1],
        "num_channels": 512, "padding": "SAME",
        "activation_fn": tf.nn.tanh
      },
      {
        "kernel_size": [5], "stride": [1],
        "num_channels": 512, "padding": "SAME",
        "activation_fn": tf.nn.tanh
      },
      {
        "kernel_size": [5], "stride": [1],
        "num_channels": -1, "padding": "SAME",
        "activation_fn": None
      }
    ],
    "mask_decoder_sequence": True,
    "parallel_iterations": 32,
  },
  
  "loss": Text2SpeechLoss,
  "loss_params": {
    "use_mask": True
  },

  "data_layer": Text2SpeechDataLayer,
  "data_layer_params": {
    "save_embeddings": False,
    "use_npy_wavs": use_npy_wavs,
    "use_phonemes": use_phonemes,
    "use_saved_embedding": use_saved_embedding,
    "saved_embedding_location": saved_embedding_location,
    
    "num_audio_features": num_audio_features,
    "output_type": output_type,
    "vocab_file": "open_seq2seq/test_utils/vocab_tts.txt",
    'dataset_location':dataset_location,
    "mag_power": 1,
    "pad_EOS": True,
    "feature_normalize": False,
    "feature_normalize_mean": 0.,
    "feature_normalize_std": 1.,
    "data_min":data_min,
    "mel_type":'htk',
    "trim": trim,   
    "duration_max":1024,
    "duration_min":24,
    "exp_mag": exp_mag
  },
}

train_params = {
  "data_layer_params": {
    "dataset_files": [
      os.path.join(dataset_location, train),
    ],
    "shuffle": True,
    "style_input": "wav"
  },
}

eval_params = {
  "data_layer_params": {
    "dataset_files": [
      os.path.join(dataset_location, val),
    ],
    "duration_max":10000,
    "duration_min":0,
    "shuffle": True,
    "style_input": "wav"
  },
}

infer_params = {
  "data_layer_params": {
    "dataset_files": [
      os.path.join(dataset_location, infer),
    ],
    "duration_max":10000,
    "duration_min":0,
    "shuffle": False,
    "style_input": "wav"
  },
}
