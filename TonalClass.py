#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified on May, 2019 by Yu Shiu

Created on Mon Dec 11 10:57:42 2017

@author: kpalmer

' tonal class is a file that contains instances of whistles
' define a filename (*.bin or *.ton) as a class of tonals
' open the header and iterate through using __next__ to return the whistles
"""
import numpy as np
import pandas as pd
import os

from datainputstream import DataInputStream
# Change directory to location of test files

# build a whistle contour class that uses polyfit, accepts the key dictionary
# import numpy.polynomial.polynomial as poly
# import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, UnivariateSpline
# from scipy.signal import spectrogram

soundfile_avail = True
_default_reader = "SoundFile"


def nearest_to(array, value):
    # Function for finding nearest neighbour
    idx = (np.abs(array-value)).argmin()
    return array[idx], idx


class TonalHeader(object):
    # initial header string, what it should say if there is a proper header
    HEADER_STR = b"silbido!"

    # Constant set to 3
    DET_VERSION = 3

    # construct the bitmask for each of the feature columns
    # t=1, f=2, snr=3 and so on
    TIME = 1
    FREQ = 1 << 1  # 2 because we are in binary
    SNR = 1 << 2
    PHASE = 1 << 3
    SCORE = 1 << 4
    CONFIDENCE = 1 << 5
    RIDGE = 1 << 6
    # Default bitmask indicating variables present
    DEFAULT = TIME | FREQ

    def __init__(self, ton_filename, ):
        """init((filename) - construct silbidio tonal header reader"""

        # Define filename
        self.ton_filename = ton_filename

        # number of bytes to read in to check if header is present
        self.magicLen = len(self.HEADER_STR)

        # load the file
        self.binary = open(self.ton_filename, "rb")

        # data input stream
        self.datainstream = DataInputStream(self.binary)

        # this is a function...
        self.ReadHeader()

    def ReadHeader(self):
        # Read in the first 8 bytes of the file-

        headerlabel = self.binary.read(self.magicLen)

        # If there is a header, set things up appropriately
        # pdb.set_trace()
        if headerlabel == self.HEADER_STR:

            # set up stream reader and then read appropriate sizes
            self.version =  self.datainstream.read_short()  # Use right one
            self.bitMask =  self.datainstream.read_short()
            self.userVersion = self.datainstream.read_short()
            self.headerSize = self.datainstream.read_int()

            # Figure out how much of the header has already been read
            self.headerused = 2 + 2 + 2 + 4 + self.magicLen # Length read in up till now in bytes

                ## Figure out how long the user comments must be
            commentLen = self.headerSize - self.headerused

            if (commentLen > 0):
                self.comment = self.datainstream.read_utf()
            else:
                self.comment = ''

        else: # no header
            self.bitMask = self.DEFAULT
            # set pointer back to byte 0 from datainpustream modification
            self.binary.seek(0)
            #print(self.bitMask)
            pass

    def hasSNR(self):
        return bool((self.bitMask & self.SNR) > 0)

    def hasPHASE(self):
        return bool((self.bitMask & self.PHASE) > 0)

    def hasRIDGE(self):
        return bool((self.bitMask & self.RIDGE) > 0)

    def hasFREQ(self):
        return bool((self.bitMask & self.FREQ) > 0)

    def hasTIME(self):
        return bool((self.bitMask & self.TIME) > 0)

    def hasCONFIDENCE(self):
        return bool((self.bitMask & self.CONFIDENCE) > 0)

    def hasSCORE(self):
        return bool((self.bitMask & self.SCORE) > 0)

    # get some things
    def getComment(self):
        return str(self.comment)

    def getUserVersion(self):
        return self.userVersion

    # COMMENTS REQIRED
    def getDatainstream(self):
        "getDataInstream() - Return DataInputStream that accesses file"
        return self.datainstream

    """
    def getFileFormatVersion(self):
        return self.comment
    """
    def getMask(self):
        return self.bitMask


class tonal(object):
    # Initialize values
    def __init__(self, fname, verbose=False):
        "__init__(filename, debug)"
        
        self.verbose = verbose
        self.whistle_idx = 0  # keep track of current whistle
                
        self.fname = fname
        self.hdr = TonalHeader(fname) # use the tonal header
        self.binary = open(self.fname, "rb") # open the binary file

        # set up the optional variables
        # SNR
        
        # Measurements, if in the file, will appear in this order.
        self.measurements = ["Time","Freq","SNR","Phase", 
                             "Score", "Confidence", "Ridge"]
        self.measurement_types = {
                "Time": ('d', self.hdr.hasTIME()),
                "Freq": ('d', self.hdr.hasFREQ()),
                "SNR": ('d', self.hdr.hasSNR()),
                "Phase": ('d', self.hdr.hasPHASE()),
                "Score": ('d', self.hdr.hasSCORE()),
                "Confidence" : ('i', self.hdr.hasCONFIDENCE()),
                "Ridge": ('i', self.hdr.hasRIDGE())
                }
        
        # Define data input stream
        self.bis = self.hdr.getDatainstream()                       
        
        # Set read format for whistle time-frequency nodes
        self.measured = [meas for meas in self.measurements
                             if self.measurement_types[meas][1] == True]
        node_fmt_list = [self.measurement_types[meas][0] 
                         for meas in self.measured]
        self.time_freq_fmt = "".join(node_fmt_list)
        
        # Read the singletons and remove from read index
        if self.hdr.hasCONFIDENCE():
            self.time_freq_fmt = self.time_freq_fmt.replace("d", "",1)
            self.measured.remove('Confidence')
            
        if self.hdr.hasSCORE():
            self.time_freq_fmt = self.time_freq_fmt.replace("d", "",1)
            self.measured.remove('Score')
            
        if self.hdr.hasSNR():
            self.time_freq_fmt = self.time_freq_fmt.replace("d", "",1)
            self.measured.remove('SNR')

        '''    
        # Check to see whether there is a header and offset by the number of bytes in the header
        # if the bitmask is equal to 3  then only time and frequency were provided and
        # no offset??
        
        if self.hdr.getMask() == 3:
            # Read in the files! 
        else:
            print('Header present, you are stuffed')
        break
            # offset by the length of the header and then read the files!
            
        '''
    
    # Define tonal as an iteratable object - THERE ARE THINGS IN HERE YOU CAN
    # ITERATE
    def __iter__(self):
        "iter(obj) - Return self as we know how to iterate"
        return self
    
    def __next__(self): 
        'next() - Return next whistle'
        Whistle_contour = dict()
        # Read in single values for the whistle
        if self.verbose:
            print("Reading whistle {} in file {}".format(
                self.whistle_idx, self.fname), end="")
        
        try:
            # Read the singletons and remove from read index
            if self.hdr.hasCONFIDENCE():
                conf = self.bis.read_double()
                Whistle_contour.update({'Confidence': conf})
                
            if self.hdr.hasSCORE():
                score = self.bis.read_double()
                Whistle_contour.update({'Score': score})
                
            if self.hdr.hasSNR():
                SNR = self.bis.read_double()
                Whistle_contour.update({'SNR': SNR})

            NumNodes = self.bis.read_int()
        except EOFError:
            raise StopIteration   # No more whistles in file
        
        if self.verbose:
            print(", {} nodes".format(NumNodes))

        # Read in the record. If only time/frequency read in all whistles at
        # once
        data = self.bis.read_record(format = self.time_freq_fmt, n=NumNodes)
        
        # Throw a warning if whistle has noting in it
        if len(data) < 1:
            print_msg = 'Problem with ' + \
                os.path.split(os.path.split(self.fname)[0])[1] + ' ' + \
                os.path.split(self.fname)[1] + ' no data read'
            
            print(print_msg)
        
        n_metrics = len(self.time_freq_fmt)
        
        for ii in range(n_metrics):
            Whistle_contour.update({str(self.measured[ii]) : data[ii::n_metrics]})

        self.whistle_idx += 1
        if len(Whistle_contour['Time'])<1:
            aa = self.fname + 'error!'
            print(aa)
             
        return Whistle_contour
    
    # getters
    def getFname(self):
        return str(self.fname)
    def getTime(self):
        return  np.array(self.Time)
    def getFreq(self):
        return  np.array(self.Freq)
    def getSNR(self):
        self.hdr.hasSNR()
        print('Tonal has no SNR values')
    def getPhase(self):
        self.hdr.hasPHASE()
        print('Tonal has no Phase values')
    def getScore(self):
        self.hdr.hasSCORE()
        print('Tonal has no Score values')
    def getConf(self):
        self.hdr.hasCONFIDENCE()
        print('Tonal has no confidence values')
    def getRidge(self):
        self.hdr.hasRIDGE()
        print('Tonal has no Ridge values')

''''
Section 2
Create a whistle contour class that will return a polynomial fit
' Part 3 : Create a class that produces training data for the NN'
'''
# tonals = tonal('/cache/kpalmer/quick_ssd/data/dclmmpa2013/'+\
#                             'LogsWithNewSNR/Silbido/NOPPSet1/NOPP6_20090329_RW_upcalls.ann')

# iter(tonals)
# whistle = next(tonals)
# whistle = next(tonals)


##############################################
## debugging #

# use ssd as data directory (speed)
# test all files
#data_dir = '/cache/kpalmer/quick_ssd/data/dclmmpa2011/devel_data/'
#data_dir = '/home/kpalmer/AnacondaProjects/data/dclmmpa2011/devel_data'
#
#data_by_species = dict()
#for species in ['bottlenose', 'common', 'melon-headed', 'spinner']:
#    data_by_species[species] = os.path.join(data_dir, species)
#
#file_dirs = list(data_by_species.values())
#
#counter = 0
## Test all files work
#for ff in file_dirs:
#    print(ff)
#    for f in os.listdir(ff):
#         if f.endswith('.bin') or f.endswith('.ton'):
#            fname = ff  + '/' + f
#            counter +=1
#            tonal_temp = tonal(fname)
#            for w in tonal_temp:
#                if len(w['Time'])<1:
#                    print('error ' + tonal_temp.getFname())
#        
#
#
#
#

#        
#prob_file ='/cache/kpalmer/quick_ssd/data/dclmmpa2011/devel_data/bottlenose/Qx-Tt-SCI0608-N1-060814-123433.bin'
#tonal_temp = tonal(prob_file, verbose=False)
#        
#    print({}.format(len(w['Time'])))
#    
#while True:
#    try:
#        w = next(tonal_temp)
#    except Exception as e:
#        break
#
#    
#    print

#%%

# From Marie
def get_corpus(dir, filetype=".wav"):
    """get_corpus(dir, filetype=".wav"
    Traverse a directory's subtree picking up all files of correct type
    """
    
    files = []
    
    # Standard traversal with os.walk, see library docs
    for dirpath, dirnames, filenames in os.walk(dir):
        for filename in [f for f in filenames if f.endswith(filetype)]:
            files.append(os.path.join(dirpath, filename))
                         
    return files


def make_whistle_df(bin_file):
    '''
    Creates a pandas datframe listing all whistle times
    
    inputs :
    bin_file - path and name of binary whistle file
    
    outputs :
    whistle_df- dictionary of all whistles in file with start and end time
    
    '''
    import pandas as pd
    
    whistle_df = pd.DataFrame(columns = ['ID', 'Start_s', 'End_s', 'Low_f', 'High_f'])
    
    # identify the tonal file 
    WhistleFile = tonal(bin_file)          
    counter = 0
    while True:
        try:
            # Read the whistle then populate a data frame
            whistle = next(WhistleFile)
            whistle_df.loc[counter] = [counter, whistle['Time'][0], 
                              whistle['Time'][-1], min(whistle['Freq']),
                              max(whistle['Freq'])]
            #print(counter)
            counter +=1
        except StopIteration:
            break
           
    # Populate properties of whistle
    whistle_df['Duration'] = whistle_df['End_s']-whistle_df['Start_s'] 
    whistle_df['Bandwidth'] = whistle_df['High_f']-whistle_df['Low_f'] 
    whistle_df = whistle_df.sort_values(['Start_s'])
    whistle_df['difftime'] = np.append([0], [np.array(whistle_df['Start_s'][1:,])- np.array(whistle_df['End_s'][0:-1])])
    whistle_df['MultWhistles'] = whistle_df['difftime']<0
    whistle_df['Spp'] =os.path.split(os.path.split(bin_file)[0])[1]
    whistle_df['file'] =os.path.split(bin_file)[1]

    return whistle_df, counter
    
    

def GetWhistelIds(whistel_df, t_start = 7, t_stop=8):
    ' Returns the indexes of the whistles that fall within t_start and t_stop'
    overlap_idx = np.where((whistel_df['Start_s'] >= t_start) & 
                           (whistel_df['End_s'] <=t_stop))[0]
    return(overlap_idx)
    

def MakeBinaryMask_chunk(bin_file, whistle_df, audio,
                         start_s, stop_s, 
                         adv_ms =2, len_ms=8,
                         OnlyComplete= True,
                         chunk_dur =None,
                         verbose = False):
    ''' Returns a binary file mask from time start_s to stop_s
    input:
        bin_file - binary file name and path
        whistle_df - dataframe of whistle times created using MakeWhistleDf
        audio - FrameStreamer (AudioFrames) of the sound file
        start_s - start time of the sound file of interest
        stop_s - end time for the sound file of interest
        adv_ms = frame advance in ms
        len_ms = frame length in ms
        verbose = True or False  (True prints information about tonals)
    '''
 
    Fs = audio.Fs
    T = (stop_s-start_s)
    
    frame_len = audio.get_framelen_samples()
    dft_bins = frame_len
    bins_Nyquist = np.floor(dft_bins/2)
    ff = np.arange(np.floor(bins_Nyquist)) / bins_Nyquist * audio.get_Nyquist()
    f_bins =len(ff)   
    
    
    t_bins = T/(adv_ms/1000)  
    tt = np.arange(start_s, start_s+T, adv_ms/1000)
    binmask = np.zeros([int(f_bins), int(t_bins)])

    # populate the whistles on the mask
    WhistleFile = tonal(bin_file, verbose=verbose)
    whistle_ids = GetWhistelIds(whistle_df, start_s, stop_s)
    
    # Create output labels
    # iterate through the binary file and project whistles where approperiate
    counter = 0
    while True:        
        try:            
            # load the whistle
            whistle_dat = next(WhistleFile)
            
            
            
             # If the whistle is in the counter then create a predictive spline
            if (counter in whistle_ids):
                t_idx = np.where((tt>min(whistle_dat['Time']))*(tt<max(whistle_dat['Time'])))[0]
                whistle_cs = interp1d(whistle_dat['Time'], whistle_dat['Freq'], kind ='cubic')
                
                for ii in range(len(t_idx)-1):
                    f = whistle_cs(np.linspace(tt[t_idx[ii]], tt[t_idx[ii+1]], 5))
                    fpred = list(map(lambda x: np.abs(ff-x).argmin(), f))
                    binmask[fpred, t_idx[ii]] = 1 

            counter +=1
        except StopIteration:
            break
        except Exception as err:
            err
            break
    # idiot checking
    #plt.pcolormesh(tt, ff[1:200], binmask[1:200,:], cmap='gray_r') 
    
    binmask = np.transpose(binmask)
    
    if OnlyComplete:
        max_frames = get_complete_frames(audio, adv_ms, len_ms, chunk_dur)
        binmask = binmask[0:max_frames,:] 
    
    return(binmask)
    


# assign whistls to groups where the minimum spacking between groups is 
# 10 sec

def MakeExampleList(whistle_df, audio, buffer_s=10, max_example_dur = None):
    '''Audio is an audio frames object (audio = AudioFrames(filename, 2,8))
        whistle_df is the datafram containin all whistle start and stop times
        buffer_s is the buffer lenght in seconds around whistles to create a
        new example
    '''
    group_id =0
    whistle_df['groupId'] = 0 
    example_list_start = []
    example_list_end = []
            
    # group the whistles into encounters
    for ii in range(whistle_df.shape[0]):
        if whistle_df['difftime'][ii]<buffer_s:
            whistle_df.loc[ii,'groupId'] = group_id
        else:
            group_id += 1
            whistle_df.loc[ii,'groupId'] = group_id
    
    # Add first example period
    if whistle_df['Start_s'].iloc[0]>buffer_s:
        example_list_start.append(0)
        example_list_end.append(whistle_df['Start_s'][0]-buffer_s)  
    
    # create time index for each encounter
    for jj in np.unique(whistle_df['groupId']):
        ids = np.where(whistle_df['groupId'] == np.unique(whistle_df['groupId'])[jj])
        ids = ids[0].tolist()
        example = [min(whistle_df['Start_s'][ids])-buffer_s, 
                   max(whistle_df['End_s'][ids])+buffer_s]
        
        # force all values to be within the time of the audio file (0:T) 
        if example[0]<0:
            example[0] = 0
        if example[1]> audio.soundfileinfo.duration:
            example[1] =audio.soundfileinfo.duration
        example_list_end.append(example[1])
        example_list_start.append(example[0])
    
 
    # Add last eperiod
    if whistle_df['End_s'].iloc[-1]+buffer_s <  audio.soundfileinfo.duration:
        example_list_end.append(audio.soundfileinfo.duration)
        example_list_start.append(whistle_df['End_s'].iloc[-1]+buffer_s)
    
    example_durations = np.subtract(example_list_end, example_list_start)    
    example_times = pd.DataFrame({'Start':example_list_start, 'End': example_list_end})
    example_times['Duration']=example_times['End']-example_times['Start']
    
#    # chunk things up smaller if maximum example duration is defined
#    if max_example_dur is not None:
#        
#        idx = np.where(example_durations > max_example_dur)[0].tolist()
#        for k in range(len(idx)):
#            
#
    return(example_times)

def get_complete_frames(audio, adv_ms, len_ms, chunk_dur = None, Offset = 0):
    '''
    Return the index of the last complete frame 
    Input - audio: AudFrames object 
            adv_ms: frame advance in ms
            len_ms: frame length in ms
            optional
            Offset= ???
            Chunk_duration = (seconds) if chunking
    '''
    SampleCount = audio.soundfileinfo.frames
    FrameShift = audio.get_frameadv_samples()
    FrameLength = audio.get_framelen_samples()
    FrameLastComplete = int(np.floor((SampleCount - Offset 
                                      - FrameLength + FrameShift) / FrameShift))
    
    if chunk_dur is not None:
        SampleCount = audio.Fs * chunk_dur
        FrameLastComplete = int(np.floor((SampleCount - Offset - 
                                          FrameLength + FrameShift) / FrameShift))

    return(FrameLastComplete)


def get_feature_labels(whislte_file, sound_file, adv_ms, len_ms, 
                       chunk_dur_s = None, log_handle=None):
    '''
    Creates list of exaples matching the size and shape of the get features
    returns either a single list of the file/size binary mask or
    - if chunk_dur_s >0 reurns list of examples
    whistlefile- binary or tonal file name/loc matching the whitsle file
    sound_file - filename/loc of the associated sound file
    adv_ms, len_ms, - advance and length
    '''
    #print('Making Mask' + sound_file)
    audio = AudioFrames(sound_file, adv_ms, len_ms,  incomplete=False)
    
    # Get whistle dataframe
    whistle_df = make_whistle_df(whislte_file)[0]
    
    # binary mask to list to match format of features
    examples_labels = MakeBinaryMask_chunk(whislte_file, whistle_df, audio,
                                      start_s = 0, stop_s=  audio.soundfileinfo.duration,
                                      adv_ms = adv_ms, len_ms=len_ms)    
    if chunk_dur_s is None:
        out_value = examples_labels
        print('No Whistle Contours')
    else:
#        examples_labels = examples_labels.tolist()
        n_subexamples = np.ceil(audio.soundfileinfo.duration/chunk_dur_s).astype(int)
        
        # number of slices in each sub example
        n_rows = int(chunk_dur_s * 1.0 / (adv_ms/1000))
        Example_subs = []
        
        # divide the labs into sub examples
        for ii in range(n_subexamples): 
            start = n_rows * ii
            stop = (n_rows * (ii + 1))-1
#           chunk = labs[start:stop]
            
            # Handeling/padding last chunk
            if ii == (n_subexamples-1):
                # padd the sample
                mm = examples_labels[start:stop,:]                
                out = np.concatenate((mm, np.zeros(((n_rows - mm.shape[0]-1),examples_labels.shape[1]))))
                Example_subs.append(out)


            else:
                # no padding needed
                Example_subs.append(examples_labels[start:stop,:])
            
        out_value = Example_subs
        
     
    #print('Mask Complete' + sound_file)    
    return(out_value)
            
        
            
    



def get_features(sound_file, adv_ms, len_ms, chunk_dur_s = None, log_handle=None):
    """get_features(file, adv_ms, len_ms, pca, components, offset_s, flatten=True)
    
    Given a file path (file), compute a spectrogram with
    framing parameters of adv_ms, len_ms.  To remove frames
    use vad or offset_s (see below)

    If a pca object is given, reduce the dimensionality of the spectra to the
    specified number of components using a PCA analysis (dsp.PCA object in
    variable pca).
     
    # Arguments
    file - Audio file to read
    adv_ms - frame advance in ms
    len_ms - frame length in ms
    chunk_dur_s - The duration of each example file, initally set to 30seconds
    more fiddiling to come

    log_handle - If present, is a handle to a file stream.  The file's
        name will be logged along with the start and end time used
        in seconds
    """
    print('Collecting Features' + sound_file)
    
    framestream = AudioFrames(sound_file, adv_ms, len_ms)
    dftstream = DFTStream(framestream)
    
    spectra = []
    for s in dftstream:
        # s is a tuple (spectrum, time offset s, timestamp)
        spectra.append(s[0])
    

    # Add extra zeros if chunk size is present
    if chunk_dur_s is not None:
        # number of chunks in total 
        n_subexamples = np.ceil(framestream.soundfileinfo.duration/chunk_dur_s).astype(int)
        
        # number of slices in each sub example
        n_rows = int(chunk_dur_s * 1.0 / (adv_ms/1000))
        
        # number of dft 0 samples to append
        n_dft_to_add = n_rows - len(spectra) % (n_rows)
        
        dummy = spectra[0]
        dummy= np.subtract(dummy, dummy)
        
        for ii in range(n_dft_to_add):
            spectra.append(dummy)
        
        # Row oriented spectra
        spectra = np.asarray(spectra)
        frames = spectra.shape[0]  # Number of spectral frames
    
        # Create subset of examples
        Example_subs = []
        
        # divide the labs into sub examples
        for ii in range(n_subexamples): 
            start = n_rows * ii
            stop = (n_rows * (ii + 1))-1
            Example_subs.append(spectra[start:stop,:])
        out_value = Example_subs
        print('Returned list of '+ repr(n_subexamples) +' examples ' + repr(chunk_dur_s) + 's duration')


    else:
        # Row oriented spectra
        spectra = np.asarray(spectra)    
        out_value = spectra       
    
        print('Returned one example ' + repr(framestream.soundfileinfo.duration) + 's duration')
    
    return out_value






    
    

def select_matching_files(fnames_wavs):
    ''' returns lists of wav and binary/tonal files where both are present 
    fnames_waves: list of wave files obtained form get_corpus
    '''
    import os
    good_wav_files=[]
    good_bin_files =[]
    dir_name =os.path.split(fnames_wavs[0])[0]
    
    for ii in range(len(fnames_wavs)):
        #range(len(fnames_wavs)):
        fname_wave = os.path.splitext(os.path.split(fnames_wavs[ii])[1])[0]
    
        if os.path.isfile(dir_name + '/' + fname_wave + '.bin'):
            bin_fname = dir_name + '/' + fname_wave + '.bin'
# TONAL FILES NOT WORKING REMOVED  
#        elif os.path.isfile(dir_name + '/' + fname_wave + '.ton'):
#            bin_fname = dir_name + '/' + fname_wave + '.ton'
        else:
            continue
        
        # List of sound files with matching binary or tonal
        good_wav_files.append(fnames_wavs[ii])
        good_bin_files.append(bin_fname)
    
    return(good_wav_files, good_bin_files)







#whistle = whistle_contour(next(tonal_temp))
#whistle_timedf.update({'ID': ii, 'Start':whistle.gettime()[0],
#                       'End':whistle.gettime()[-1], 'Fname':fname})    
class MakeBinMask():
    '''
    inputs :
    bin_file - path and name of binary whistle file
    sound_file - path and name of soundfile
    adv_ms - spectrogram advance in miliseconds
    len_ms - spectrum length in miliseconds
    Returns:
    self.whistlemasek-binary mask for the soundfile in question with
    associated whistles 
    self.binmask the empty binary mask size of the spectrogram for the whole 
    file
    
    '''
    def __init__(self, bin_file, sound_file, adv_ms, len_ms):
        
        # Store params
        self.bin_file = bin_file
        self.sound_file = sound_file
        self.adv_ms = adv_ms
        self.len_ms = len_ms
        self.fileobj = SoundFile(self.sound_file)
        self.Fs = self.fileobj.samplerate
        extra_info = self.fileobj.extra_info
        self.samplesN = int(extra_info[extra_info.find('data :')+7: len(extra_info)-5])
        self.channels = self.fileobj.channels
        self.format = self.fileobj.format

        
        # Create the binary mask
        self.binmask, self.ff, self.tt = self.make_empty_binmask()
        self.whistlemask = self.projectWhistles()

    def make_empty_binmask(self):
        '''
        Returns an ampty array of zeros the width and height of the sound
        file
        '''
        T = 1/self.Fs * self.samplesN

        frame_len = audio.get_framelen_samples()
        dft_bins = frame_len
        bins_Nyquist = np.floor(dft_bins/2)
        ff = np.arange(np.floor(bins_Nyquist)) / bins_Nyquist * audio.get_Nyquist()
        f_bins =len(ff)           
        
        t_bins = T/(self.adv_ms/1000)  
        tt = np.linspace(0, T, round(t_bins))
        
        binmask = np.zeros([round(f_bins), round(t_bins)])
        return(binmask, ff, tt)
    
    def projectWhistles(self):
       
        dsp_params = dict({'fs':self.Fs,
                               'pad_dur':0,
                               'nfft':self.Fs * self.len_ms/1000,
                               'noverlap':self.Fs * self.adv_ms/1000,
                               'soundfilename': None})
        
        WhistleFile = tonal(self.bin_file, dsp_params)          
        counter = 0
        while True:
            counter +=1
            try:
                # load the whistle
                whistle = whistle_contour(next(WhistleFile), dsp_params)
                
                t_idx = np.where((self.tt>min(whistle.Time))*(self.tt<max(whistle.Time)))[0]
                
                for ii in range(len(t_idx)-1):
                    f = whistle.cs(np.linspace(tt[t_idx[ii]], tt[t_idx[ii+1]], 5))
                    fpred = list(map(lambda x: np.abs(ff-x).argmin(), f))
                    binmask[fpred, t_idx[ii]] = 1   
            except StopIteration:
                break
            
            return(binmask)
        
        idx = (np.abs(array-value)).argmin()
        
        













