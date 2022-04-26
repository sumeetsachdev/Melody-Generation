import os
import json
import numpy as np
import music21 as m21
import keras

MIDI_DATASET_PATH = "pink_floyd_midi"
ACCEPTABLE_DURATIONS = [0.25, 0.5, 0.75, 1.0, 1.5, 2, 3, 4]
SAVE_DIR = 'dataset'
SINGLE_FILE_DATASET = 'file_dataset'
MAPPING_PATH = 'mapping.json'
SEQUENCE_LENGTH = 64


def load_songs_in_midi(dataset_path):

    songs = []
    
    # go through all the files in dataset and load them with music21
    for path, subdirs, files in os.walk(dataset_path):
        for file in files:
            if file[-3:] == 'mid':
                song_name = os.path.join(path, file)
                mf = m21.midi.MidiFile()
                mf.open(str(song_name))
                mf.read()
                mf.close()
##                print(len(mf.tracks))
                song = m21.midi.translate.midiFileToStream(mf)
##                song = m21.converter.parse(song_name)
##                print(type(song))
##                song = m21.midi.translate.midiFileToStream(song_name)
##                print(type(song))
                songs.append(song)
    return songs


def has_acceptable_durations(song, acceptable_durations):
    for note in song.flat.notesAndRests:
        if note.duration.quarterLength not in acceptable_durations:
            return False
    return True

def transpose(song):
##    parts = song.getElementsByClass(m21.stream.Part)
##    measure_part0 = parts[0].getElementsByClass(m21.stream.Measure)
##    key = measure_part0[0][4]
##
##
##    if not isinstance(key, m21.key.Key):
##        key = song.analyze("key")

    key = song.analyze('key')

    if key.mode == 'major':
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch('C'))
    elif key.mode == 'minor':
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch('A'))


    transposed_song = song.transpose(interval)

    return transposed_song


def encode_song(song, time_step=0.25):
    encoded_song = []
    for event in song.flat.notesAndRests:
        if isinstance(event, m21.note.Note):
            symbol = event.pitch.midi
##        else isinstance(event, m21.note.Rest):
        else:
            symbol = 'r'

        steps = int(event.duration.quarterLength / time_step)
        for step in range(steps):
            if step == 0:
                encoded_song.append(symbol)
            else:
                encoded_song.append('_')
              
    encoded_song = " ".join(map(str, encoded_song))
    return encoded_song

def preprocess(dataset_path):

    # load the folk songs
    print("Loading songs...")
    songs = load_songs_in_midi(dataset_path)
    print(f'Loaded {len(songs)} songs.')

    for i,song in enumerate(songs):

        # filter out songs that have non-acceptable durations
        """
        if not has_acceptable_durations(song, ACCEPTABLE_DURATIONS):
            continue
        """
        song = transpose(song)
        encoded_song = encode_song(song)
        save_path = os.path.join(SAVE_DIR, str(i))
        with open(save_path, 'w') as fp:
            fp.write(encoded_song)

def load(file_path):
    with open(file_path, 'r') as fp:
        song = fp.read()
    return song

def create_single_file_dataset(dataset_path, file_dataset_path, sequence_length):
    new_song_delimiter = "/ " * sequence_length
    songs = ""

    for path, _, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(path, file)
            song = load(file_path)
            songs = songs + song + " " + new_song_delimiter
    songs = songs[:-1]

    with open(file_dataset_path, 'w') as fp:
        fp.write(songs)
    return songs

def create_mapping(songs, mapping_path):
    mappings = {}

    songs = songs.split()
    vocabulary = list(set(songs))
##    print(len(vocabulary))

    for i, symbol in enumerate(vocabulary):
        mappings[symbol] = i

    with open(mapping_path, 'w') as fp:
        json.dump(mappings, fp)


def convert_songs_to_int(songs):
    int_songs = []

    with open(MAPPING_PATH, 'r') as fp:
        mappings = json.load(fp)

    songs = songs.split()

    for symbol in songs:
        int_songs.append(mappings[symbol])

    return int_songs


def generate_training_sequences(sequence_length):
    songs = load(SINGLE_FILE_DATASET)
    int_songs = convert_songs_to_int(songs)

    inputs, targets = [], []

    num_sequences = len(int_songs) - sequence_length
    for i in range(num_sequences):
        inputs.append(int_songs[i:i+sequence_length])
        targets.append(int_songs[i + sequence_length])

##    vocabulary_size = len(set(int_songs))
##    print(inputs[4])
##    inputs = keras.utils.to_categorical(inputs, num_classes=vocabulary_size)
##    targets = np.array(targets)

    return inputs, targets
    
    


  
if __name__ == '__main__':
##    preprocess(MIDI_DATASET_PATH)
    songs = create_single_file_dataset(SAVE_DIR, SINGLE_FILE_DATASET, SEQUENCE_LENGTH)
    create_mapping(songs, MAPPING_PATH)
    inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)
    
    
