Speech Accent Detection
==============================

Human speaks a language with an accent. A particular accent essentially reflects a person's linguistic background. The model define accent based audio record. The result of the model could be used to define accent and help to decrease accent to English learning students and improve accent by training.


## Outline
<img src="img/speech_vovels.jpeg" width="900">

- [About](#about)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Mel Frequency Cepstrum Coefficients](#mel-frequency-cepstrum-coefficients--mfcc-)
- [Overview](#overview-1)
- [Models](#models)
    - [FFNN](#ffnn)
    - [CNN](#cnn)
    - [LSTM](#lstm) 
- [Incorrect Classifications](#incorrect-classifications)
- [Future Work](#future-work)


### About

English language is global language, it is became as a must language for the most people. Since english language going to be part of different nationals, the origin english language is changing based location and the mother languages of the people in that area. So on we can find American-English, England-English, Indian-English and other english languages. One the significant different between different english languages is accent. 

Accent detection would allow to define accent level of the student and re-train with native speaker or for the school that want to hire the teacher, could define accent on the teacher. 

### Objective
+ The model that can classify the accent of the speaker based on the audio file (wav format).

### Requirements
Model run on Ubuntu 18.04
Python requirement libraries in requirements.txt
Moreover needs:
- ffmpeg
- cuda
- 

### Dataset
1. [George Mason University Speech Accent Archive](http://accent.gmu.edu/about.php) dataset contains around 3500 audio files and speakers from over 100 countries.
    
    All speakers in the dataset read from the same passage:
    >  "Please call Stella. Ask her to bring these things with her from the store:  Six spoons of fresh snow peas, five thick slabs of blue cheese, and maybe a snack for her brother Bob.  We also need a small plastic snake and a big toy frog for the kids.  She can scoop these things into three red bags, and we will go meet her Wednesday at the train station."

2. [CSTR VCTK Corpus: English Multi-speaker Corpus for CSTR Voice Cloning Toolkit (version 0.92)](https://datashare.is.ed.ac.uk/handle/10283/3443).

    This CSTR VCTK Corpus includes speech data uttered by 110 English speakers with various accents. Each speaker reads out about 400 sentences, which were selected from a newspaper, the rainbow passage and an elicitation paragraph used for the speech accent archive. The newspaper texts were taken from Herald Glasgow, with permission from Herald & Times Group. Each speaker has a different set of the newspaper texts selected based a greedy algorithm that increases the contextual and phonetic coverage. 

3. [Mozilla Voice data](https://commonvoice.mozilla.org/en/about). The Mozilla Voice data contains tens of thousands of files of native and non-native speakers speaking different sentences. Because the audio files are so different from speaker to speaker, I am working with the smaller, more static, Speech Accent Archive first in order to get a good working model that will identify the correct signal and then use those saved weights to train the Mozilla Voice data.

The dataset contained **.mp3** audio files which were converted to **.wav** audio files

### Mel Frequency Cepstrum Coefficients (MFCC)
To vectorize the audio files by creating [MFCC's](https://wiki.aalto.fi/display/ITSP/Cepstrum+and+MFCC). MFCC's are meant to mimic the biological process of humans creating sound to produce phonemes and the way humans perceive these sounds.

Phonemes are base units of sounds that combine to make up words for a language. Non-native English speakers will use different phonemes than native speakers. The phonemes non-native speakers use will also be unique to their native language(s). By identifying the difference in phonemes, we will be able to differentiate between accents.

### Overview
  1. **Bin the raw audio signal**  
  Better to produce a matrix from a continuous signal, by binning the audio signal. On short time scales, we assume that audio signals do not change very much. Longer frames will vary too much and shorter frames will not provide enough signal. The standard is to bin the raw audio signal into 20-40 ms frames.
 
     The following steps are applied over every single one of the frames and a set of coefficients is determined for every frame:

  2. **Calculate the periodogram power estimates**  
  This process models how the cochlea interprets sounds by vibrating at different locations based on the incoming frequencies. The periodogram is an analog for this process as it measures spectral density at different frequencies. First, we need to take the Discrete Fourier Transform of every frame. The periodogram power estimate is calculated using the following equation:   

  3. **Apply mel filterbank and sum energies in each filter**  
  The cochlea can't differentiate between frequencies that are very close to each other. This problem is amplified at higher frequencies, meaning that greater ranges of frequencies will be increasingly interpreted as the same pitch. So, we sum up the signal at various increasing ranges of frequencies to get a measure of the energy density in each range.

     This [filterbank](http://www.practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/) is a set of 26 triangular filters. These filters are vectors that are mostly zero, except for a small range of the spectrum. First, we convert frequencies to the Mel Scale (converts actual tone of a frequency to its perceived frequency). Then we multiply each filter with the power spectrum and add up the resulting coefficients in order to obtain the filterbank energies. In the end, we will have a single coefficient for each filter.

  4. **Take log of all filter energies**  
  We need to take the log of the previously calculated filterbank energies because humans can differentiate between low frequency sounds better than they can between high frequency sounds. The shape of the matrix hasn't changed, so we still have 26 coefficients.

  5. **Take Discrete Cosine Transform (DCT)** of the log filterbank energies </b>  
  Because the standard is to create overlapping filterbanks, these energies are correlated and we use DCT to decorrelate them. The higher DCT coefficients are then dropped, which has been shown to perform model performance, leaving us with 13 cepstral coefficients.

Resources for learning about MFCC's:   
1) [Pratheeksha Nair's Medium Aricle](https://medium.com/prathena/the-dummys-guide-to-mfcc-aceab2450fd)  
2) [Haytham Fayek's Personal Website](https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html)  
3) [Practical Cryptography](http://www.practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/)

## Models
Because the Mozilla Dataset has a lot of variability in terms of what the speaker is reciting, I decided to use the Speech Accent Archive Data first and save the weights from my model with the highest accuracy to train the Mozilla Dataset. This method worked very well and saved a lot of training time because my Mozilla Data would start off at ~65% validation accuracy at the first epoch.

From [Keras FAQ](https://keras.io/getting-started/faq/):
"A Keras model has two modes: training and testing. Regularization mechanisms, such as Dropout and L1/L2 weight regularization, are turned off at testing time.

Besides, the training loss is the average of the losses over each batch of training data. Because your model is changing over time, the loss over the first batches of an epoch is generally higher than over the last batches. On the other hand, the testing loss for an epoch is computed using the model as it is at the end of the epoch, resulting in a lower loss."

### Feed-Forward Neural Network (FFNN) 
The FFNN model architecture:  
<p align="center"><img src="img/ffnn_architecture.png" width="600"></p>

**Classification Results**
<p align="center"><img src="img/two_class_gmu_130_epochs.png" width="600"></p>

**Test Accuracy: 78%**

### Convolution Neural Network (CNN)
The CNN model architecture:  
<p align="center"><img src="img/cnn_architecture.png" width="600"></p>

**Classification Results**
<p align="center"><img src="img/two_class_gmu_130_epochs.png" width="600"></p>

**Test Accuracy: 78%**

### Long Short-Term Memory (LSTM)
The LSTM model architecture:  
<p align="center"><img src="img/lstm_architecture.png" width="600"></p>

**Classification Results**
<p align="center"><img src="img/two_class_gmu_130_epochs.png" width="600"></p>

**Test Accuracy: 78%**

## Incorrect Classifications
Going back and listening to the files where my model failed brought two conclusions:
 + The majority of the misclassified test data was incorrectly labeled
 + Most of the remaining misclassified data was problematic because the accent seemed to be a blend, indicating that the speaker may also be fluent in another language.

While it would be best if the model could also correctly classify these blended accents, the blended accents may not pose a serious problem because speech recognition systems may not have a problem picking up what these speakers are saying. For example, a speech recognition system trained mainly on US data may be able to pick up fairly well on a speaker who from the UK who has spent a fair amount of time in the US. As long as the model is classifying speakers with more traditional UK accents, we can build another speech recognition model for these speakers.

## Future Work

Try to classify more accents, intead of native and non-native accents. It would ultimately like to train a CNN to classify most of the accents with more dataset.

Unfortunately, dataset is not enough for multi-accent detection.

