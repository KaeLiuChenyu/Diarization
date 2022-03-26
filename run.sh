#!/usr/bin/env bash

. ./cmd.sh
. ./path.sh
set -e

stage=0
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc

# path to sph files
data_root=
# path to pre-trained x-vector model
nnet_dir=
# Number of cpus
nj=

# the number of UBM components (used for VB resegmentation)
num_components=1024 
# the dimension of i-vector (used for VB resegmentation)
ivector_dim=400 


echo "stage 0"
#Prepare callhome dataset
if [ $stage -eq 0 ]; then
  local/make_callhome.sh $data_root data
fi


echo "stage 1"
#Prepare mfcc features
if [ $stage -eq 1 ]; then
  for name in callhome1 callhome2 callhome; do
    steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj $nj \
      --cmd "$train_cmd" --write-utt2num-frames true \
      data/$name exp/make_mfcc $mfccdir
    utils/fix_data_dir.sh data/$name
  done

  for name in callhome1 callhome2; do
    sid/compute_vad_decision.sh --nj $nj --cmd "$train_cmd" \
      data/$name exp/make_vad $vaddir
    utils/fix_data_dir.sh data/$name
  done
fi


echo "stage 2"
if [ $stage -eq 2 ]; then
  # This writes features to disk after applying the sliding window CMN.
  # Although this is somewhat wasteful in terms of disk space, for diarization
  # it ends up being preferable to performing the CMN in memory.  If the CMN
  # were performed in memory (e.g., we used --apply-cmn true in
  # diarization/nnet3/xvector/extract_xvectors.sh) it would need to be
  # performed after the subsegmentation, which leads to poorer results.
  for name in callhome1 callhome2; do
    local/nnet3/xvector/prepare_feats.sh --nj $nj --cmd "$train_cmd" \
      data/$name data/${name}_cmn exp/${name}_cmn
    cp data/$name/vad.scp data/${name}_cmn/
    if [ -f data/$name/segments ]; then
      cp data/$name/segments data/${name}_cmn/
    fi
    utils/fix_data_dir.sh data/${name}_cmn
  done

fi


echo "stage 3"
# Extract x-vectors
if [ $stage -eq 3 ]; then
  # Extract x-vectors for the two partitions of callhome.
  diarization/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 5G" \
    --nj $nj --window 1.5 --period 0.75 --apply-cmn false \
    --min-segment 0.5 $nnet_dir \
    data/callhome1_cmn $nnet_dir/xvectors_callhome1

  diarization/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 5G" \
    --nj $nj --window 1.5 --period 0.75 --apply-cmn false \
    --min-segment 0.5 $nnet_dir \
    data/callhome2_cmn $nnet_dir/xvectors_callhome2  
fi


echo "stage 4"
# Perform PLDA scoring
if [ $stage -eq 4 ]; then
  # Perform PLDA scoring on all pairs of segments for each recording.
  # The first directory contains the PLDA model that used callhome2
  # to perform whitening (recall that we're treating callhome2 as a
  # held-out dataset).  The second directory contains the x-vectors
  # for callhome1.
  diarization/nnet3/xvector/score_plda.sh --cmd "$train_cmd --mem 4G" \
    --nj $nj $nnet_dir/xvectors_callhome2 $nnet_dir/xvectors_callhome1 \
    $nnet_dir/xvectors_callhome1/plda_scores

  diarization/nnet3/xvector/score_plda.sh --cmd "$train_cmd --mem 4G" \
    --nj $nj $nnet_dir/xvectors_callhome1 $nnet_dir/xvectors_callhome2 \
    $nnet_dir/xvectors_callhome2/plda_scores
fi



echo "stage 5"
# Cluster the PLDA scores using a stopping threshold.
if [ $stage -eq 5 ]; then
  # First, we find the threshold that minimizes the DER on each partition of
  # callhome.
  mkdir -p $nnet_dir/tuning
  for dataset in callhome1 callhome2; do
    echo "Tuning clustering threshold for $dataset"
    best_der=100
    best_threshold=0
    utils/filter_scp.pl -f 2 data/$dataset/wav.scp \
      data/callhome/fullref.rttm > data/$dataset/ref.rttm

    # The threshold is in terms of the log likelihood ratio provided by the
    # PLDA scores.  In a perfectly calibrated system, the threshold is 0.
    # In the following loop, we evaluate the clustering on a heldout dataset
    # (callhome1 is heldout for callhome2 and vice-versa) using some reasonable
    # thresholds for a well-calibrated system.
    for threshold in -0.3 -0.2 -0.1 -0.05 0 0.05 0.1 0.2 0.3; do
      diarization/cluster.sh --cmd "$train_cmd --mem 4G" --nj $nj \
        --threshold $threshold $nnet_dir/xvectors_$dataset/plda_scores \
        $nnet_dir/xvectors_$dataset/plda_scores_t$threshold

      md-eval.pl -1 -c 0.25 -r data/$dataset/ref.rttm \
       -s $nnet_dir/xvectors_$dataset/plda_scores_t$threshold/rttm \
       2> $nnet_dir/tuning/${dataset}_t${threshold}.log \
       > $nnet_dir/tuning/${dataset}_t${threshold}

      der=$(grep -oP 'DIARIZATION\ ERROR\ =\ \K[0-9]+([.][0-9]+)?' \
        $nnet_dir/tuning/${dataset}_t${threshold})
      if [ $(perl -e "print ($der < $best_der ? 1 : 0);") -eq 1 ]; then
        best_der=$der
        best_threshold=$threshold
      fi
    done
    echo "$best_threshold" > $nnet_dir/tuning/${dataset}_best
  done

  # Cluster callhome1 using the best threshold found for callhome2.  This way,
  # callhome2 is treated as a held-out dataset to discover a reasonable
  # stopping threshold for callhome1.
  diarization/cluster.sh --cmd "$train_cmd --mem 4G" --nj $nj \
    --threshold $(cat $nnet_dir/tuning/callhome2_best) \
    $nnet_dir/xvectors_callhome1/plda_scores $nnet_dir/xvectors_callhome1/plda_scores

  # Do the same thing for callhome2, treating callhome1 as a held-out dataset
  # to discover a stopping threshold.
  diarization/cluster.sh --cmd "$train_cmd --mem 4G" --nj $nj \
    --threshold $(cat $nnet_dir/tuning/callhome1_best) \
    $nnet_dir/xvectors_callhome2/plda_scores $nnet_dir/xvectors_callhome2/plda_scores

  mkdir -p $nnet_dir/results
  # without overlap
  cat $nnet_dir/xvectors_callhome1/plda_scores/rttm \
    $nnet_dir/xvectors_callhome2/plda_scores/rttm | md-eval.pl -1 -c 0.25 -r \
    data/callhome/fullref.rttm -s - 2> $nnet_dir/results/threshold.log \
    > $nnet_dir/results/DER_threshold.txt

  der=$(grep -oP 'DIARIZATION\ ERROR\ =\ \K[0-9]+([.][0-9]+)?' \
    $nnet_dir/results/DER_threshold.txt)
  # Using the oracle number of speakers, DER: 8.58% (no overlap)
  echo "Using supervised calibration, DER: $der% (no overlap)"

  # with overlap
  cat $nnet_dir/xvectors_callhome1/plda_scores/rttm \
    $nnet_dir/xvectors_callhome2/plda_scores/rttm | md-eval.pl -c 0.25 -r \
    data/callhome/fullref.rttm -s - 2> $nnet_dir/results/threshold.log \
    > $nnet_dir/results/DER_threshold.txt

  der=$(grep -oP 'DIARIZATION\ ERROR\ =\ \K[0-9]+([.][0-9]+)?' \
    $nnet_dir/results/DER_threshold.txt)
  # Using the oracle number of speakers, DER: 18.02% (overlap)
  echo "Using supervised calibration, DER: $der% (overlap)"
fi
