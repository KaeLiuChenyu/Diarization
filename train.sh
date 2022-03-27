#!/usr/bin/env bash

. ./cmd.sh
. ./path.sh
set -e
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc

#Path to your corpus folder
data_root=
#Number of your CPUs
nj=

stage=0
nnet_dir=exp/xvector_nnet_1a/
num_components=1024 # the number of UBM components (used for VB resegmentation)
ivector_dim=400 # the dimension of i-vector (used for VB resegmentation)

# Prepare datasets
if [ $stage -le 0 ]; then
  # Prepare a collection of NIST SRE data. This will be used to train,x-vector DNN and PLDA model.
  local/make_sre.sh $data_root data

  # Prepare SWB for x-vector DNN training.
  local/make_swbd2_phase1.pl $data_root/LDC98S75 \
                           data/swbd2_phase1_train
  local/make_swbd2_phase2.pl $data_root/LDC99S79 \
                           data/swbd2_phase2_train
  local/make_swbd2_phase3.pl $data_root/LDC2002S06 \
                           data/swbd2_phase3_train
  local/make_swbd_cellular1.pl $data_root/LDC2001S13 \
                             data/swbd_cellular1_train
  local/make_swbd_cellular2.pl $data_root/LDC2004S07 \
                             data/swbd_cellular2_train

  utils/combine_data.sh data/train \
    data/swbd_cellular1_train data/swbd_cellular2_train \
    data/swbd2_phase1_train \
    data/swbd2_phase2_train data/swbd2_phase3_train data/sre
fi

if [ $stage -le 1 ]; then
  #MFCC
  steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj $nj \
    --cmd "$train_cmd" --write-utt2num-frames true \
    data/train exp/make_mfcc $mfccdir
  utils/fix_data_dir.sh data/$name

  #VAD
  sid/compute_vad_decision.sh --nj $nj --cmd "$train_cmd" \
    data/train exp/make_vad $vaddir
  utils/fix_data_dir.sh data/$name

  # The sre dataset is a subset of train
  cp data/train/{feats,vad}.scp data/sre/
  utils/fix_data_dir.sh data/sre

  # This writes features to disk after applying the sliding window CMN.
  # Although this is somewhat wasteful in terms of disk space, for diarization
  # it ends up being preferable to performing the CMN in memory.  If the CMN
  # were performed in memory (e.g., we used --apply-cmn true in
  # diarization/nnet3/xvector/extract_xvectors.sh) it would need to be
  # performed after the subsegmentation, which leads to poorer results.
  local/nnet3/xvector/prepare_feats.sh --nj $nj --cmd "$train_cmd" \
    data/train data/train_cmn exp/train_cmn
  cp data/train/vad.scp data/train_cmn/
  if [ -f data/train/segments ]; then
    cp data/train/segments data/train_cmn/
  fi
  utils/fix_data_dir.sh data/train_cmn


  echo "0.01" > data/sre_cmn/frame_shift
  # Create segments to extract x-vectors from for PLDA training data.
  # The segments are created using an energy-based speech activity
  # detection (SAD) system, but this is not necessary.  You can replace
  # this with segments computed from your favorite SAD.
  diarization/vad_to_segments.sh --nj $nj --cmd "$train_cmd" \
    data/sre_cmn data/sre_cmn_segmented
fi

# In this section, we augment the training data with reverberation,
# noise, music, and babble, and combined it with the clean data.
# The combined list will be used to train the xvector DNN.  The SRE
# subset will be used to train the PLDA model.
if [ $stage -le 2 ]; then
  frame_shift=0.01
  awk -v frame_shift=$frame_shift '{print $1, $2*frame_shift;}' data/train/utt2num_frames > data/train/reco2dur
  if [ ! -d "RIRS_NOISES" ]; then
    # Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises
    wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
    unzip rirs_noises.zip
  fi

  # Make a version with reverberated speech
  rvb_opts=()
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")

  # Make a reverberated version of the SWBD+SRE list.  Note that we don't add any
  # additive noise here.
  steps/data/reverberate_data_dir.py \
    "${rvb_opts[@]}" \
    --speech-rvb-probability 1 \
    --pointsource-noise-addition-probability 0 \
    --isotropic-noise-addition-probability 0 \
    --num-replications 1 \
    --source-sampling-rate 8000 \
    data/train data/train_reverb
  cp data/train/vad.scp data/train_reverb/
  utils/copy_data_dir.sh --utt-suffix "-reverb" data/train_reverb data/train_reverb.new
  rm -rf data/train_reverb
  mv data/train_reverb.new data/train_reverb

  # Prepare the MUSAN corpus, which consists of music, speech, and noise
  # suitable for augmentation.
  steps/data/make_musan.sh --sampling-rate 8000 data_root/musan

  # Get the duration of the MUSAN recordings.  This will be used by the
  # script augment_data_dir.py.
  for name in speech noise music; do
    utils/data/get_utt2dur.sh data/musan_${name}
    mv data/musan_${name}/utt2dur data/musan_${name}/reco2dur
  done

  # Augment with musan_noise
  steps/data/augment_data_dir.py --utt-suffix "noise" --fg-interval 1 --fg-snrs "15:10:5:0" --fg-noise-dir "data/musan_noise" data/train data/train_noise
  # Augment with musan_music
  steps/data/augment_data_dir.py --utt-suffix "music" --bg-snrs "15:10:8:5" --num-bg-noises "1" --bg-noise-dir "data/musan_music" data/train data/train_music
  # Augment with musan_speech
  steps/data/augment_data_dir.py --utt-suffix "babble" --bg-snrs "20:17:15:13" --num-bg-noises "3:4:5:6:7" --bg-noise-dir "data/musan_speech" data/train data/train_babble

  # Combine reverb, noise, music, and babble into one directory.
  utils/combine_data.sh data/train_aug data/train_reverb data/train_noise data/train_music data/train_babble

  # Take a random subset of the augmentations (128k is somewhat larger than twice
  # the size of the SWBD+SRE list)
  utils/subset_data_dir.sh data/train_aug 128000 data/train_aug_128k
  utils/fix_data_dir.sh data/train_aug_128k

  # Make filterbanks for the augmented data.  Note that we do not compute a new
  # vad.scp file here.  Instead, we use the vad.scp from the clean version of
  # the list.
  steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj $nj --cmd "$train_cmd" \
    data/train_aug_128k exp/make_mfcc $mfccdir

  # Combine the clean and augmented SWBD+SRE list.  This is now roughly
  # double the size of the original clean list.
  utils/combine_data.sh data/train_combined data/train_aug_128k data/train
fi

if [ $stage -le 3 ]; then
  # This script applies CMN and removes nonspeech frames.  Note that this is somewhat
  # wasteful, as it roughly doubles the amount of training data on disk.  After
  # creating training examples, this can be removed.
  local/nnet3/xvector/prepare_feats_for_egs.sh --nj $nj --cmd "$train_cmd" \
    data/train_combined data/train_combined_cmn_no_sil exp/train_combined_cmn_no_sil
  utils/fix_data_dir.sh data/train_combined_cmn_no_sil

  # Now, we need to remove features that are too short after removing silence
  # frames.  We want atleast 5s (500 frames) per utterance.
  min_len=500
  mv data/train_combined_cmn_no_sil/utt2num_frames data/train_combined_cmn_no_sil/utt2num_frames.bak
  awk -v min_len=${min_len} '$2 > min_len {print $1, $2}' data/train_combined_cmn_no_sil/utt2num_frames.bak > data/train_combined_cmn_no_sil/utt2num_frames
  utils/filter_scp.pl data/train_combined_cmn_no_sil/utt2num_frames data/train_combined_cmn_no_sil/utt2spk > data/train_combined_cmn_no_sil/utt2spk.new
  mv data/train_combined_cmn_no_sil/utt2spk.new data/train_combined_cmn_no_sil/utt2spk
  utils/fix_data_dir.sh data/train_combined_cmn_no_sil

  # We also want several utterances per speaker. Now we'll throw out speakers
  # with fewer than 8 utterances.
  min_num_utts=8
  awk '{print $1, NF-1}' data/train_combined_cmn_no_sil/spk2utt > data/train_combined_cmn_no_sil/spk2num
  awk -v min_num_utts=${min_num_utts} '$2 >= min_num_utts {print $1, $2}' \
    data/train_combined_cmn_no_sil/spk2num | utils/filter_scp.pl - data/train_combined_cmn_no_sil/spk2utt \
    > data/train_combined_cmn_no_sil/spk2utt.new
  mv data/train_combined_cmn_no_sil/spk2utt.new data/train_combined_cmn_no_sil/spk2utt
  utils/spk2utt_to_utt2spk.pl data/train_combined_cmn_no_sil/spk2utt > data/train_combined_cmn_no_sil/utt2spk

  utils/filter_scp.pl data/train_combined_cmn_no_sil/utt2spk data/train_combined_cmn_no_sil/utt2num_frames > data/train_combined_cmn_no_sil/utt2num_frames.new
  mv data/train_combined_cmn_no_sil/utt2num_frames.new data/train_combined_cmn_no_sil/utt2num_frames

  # Now we're ready to create training examples.
  utils/fix_data_dir.sh data/train_combined_cmn_no_sil
fi

if [ $stage -le 4 ]; then
  #Train xvector extractor
  local/nnet3/xvector/tuning/run_xvector_1a.sh --stage $stage --train-stage -1 \
    --data data/train_combined_cmn_no_sil --nnet-dir $nnet_dir \
    --egs-dir $nnet_dir/egs
fi