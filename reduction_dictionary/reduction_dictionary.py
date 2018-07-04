import numpy as np

reduction_dictionary = np.load("sound_to_reduced_sound.npy").item()

test_transcription = "pʰlis kɔl stɛːlʌ ɑsk˺ ɜ tə"
print ('full transcription:', test_transcription)

reduced_transcription = "".join(list(map(lambda letter: reduction_dictionary[letter], test_transcription)))

print ('reduced transcription:', reduced_transcription)
