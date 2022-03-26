# Callhome Diarization


## Table of Contents

- [Install](#install)
    - [Kaldi](#Kaldi)
    - [Dataset](#Dataset)
    - [Pretrain model](#Pretrain_model)
- [Usage](#usage)
    - [Inference](###Inference)
    

## Install
### Kaldi
1. This project uses [Kaldi](https://github.com/kaldi-asr), which is a state-of-the-art automatic speech recognition toolkit. Download via this [link](https://github.com/kaldi-asr/kaldi.git) and unzip the file:
```sh
$ unzip kaldi-master.zip
$ cd kaldi-master
#Your <KALDI_ROOT> will be  /your/path/to/this/folder/kaldi-master
```
 * Check `<INSTALL>`, then follow the instruction to install Kaldi.

 * If you are using Google Colab, you can install Kaldi with these comand:
```sh
!pip install kora -q
import kora.install.kaldi
#Your <KALDI_ROOT> will be /opt/kaldi
```
### Dataset
1. Download Callhome sph dataset via this [link](https://drive.google.com/drive/folders/1-LlaDXJrUyM23pF7pXWimKZTwG-EzDUt?usp=sharing)
### Pretrain_model
1. Download the kaldi's pre-trained CALLHOME x-vector diarization model via this [link](https://kaldi-asr.org/models/m6) and decompress:
```sh
$ tar -xvf /your/path/to/0006_callhome_diarization_v2_1a.tar
```
2. Cope pre-trained model to project folder
```sh
$ cp -r /your/path/to/0006_callhome_diarization_v2_1a/exp/xvector_nnet_1a <KALDI_ROOT>/egs/callhome_diarization/v2/exp
```

## Usage

### Inference
```sh
#Go to project folder
$ cd /kaldi-master/egs/callhome_diarization/v2
```
1. Edit `<cmd.sh>`

 * If you have no queueing system and want to run on a local machine, you can open `<cmd.sh>` file and change 'queue.pl' to 'run.pl'.
2. Edit `<path.sh>`

 * If you have other Kaldi root, open `<path.sh>`:
```sh
export KALDI_ROOT=`pwd`/../../..
...
```
Then change `<KALDI_ROOT>` to your own Kaldi root:
```sh
export KALDI_ROOT=/your/path/to/kaldi-master
...
```
3. Edit `<run.sh>`
 * Replace `<run.sh>` with `<run.sh>` in this report:
```sh
$ cp -rf /path/to/this/repo/run.sh <KALDI_ROOT>/egs/callhome_diarization/v2/run.sh
```
 * Makding following chages to the top of the script:
```sh
data_root=/path/containing/the/sph/files/of/CALLHOME
nnet_dir=/path/to/kaldi/xvector/model/downloaded/xvector_nnet_1a
nj=Number of your CPUs
```
4. Execute the `<run.sh>` diarization recipe step-by-step as shown below:
```sh
#Repeat until 5th stage in run.sh has been executed:
$ ./run.sh
#Now increment the stage variable on top of run.sh by 1
```
 * You are supposed to get these results:
```sh
Using supervised calibration, DER: 8.58% (no overlap)
Using supervised calibration, DER: 18.02% (overlap)
```
## License

[MIT](LICENSE) © Richard Littauer
