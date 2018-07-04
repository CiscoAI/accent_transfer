import re
import os
import stress
import difflib as dl

from tqdm import tqdm


class Sound:
    def __init__(self, sound, is_first):
        self.value = sound
        self.occurences = 1
        if is_first:
            self.was_first_sound = 1
        else:
            self.was_first_sound = 0
        self.replacings = {}
        self.first_replacings = {}
        self.insertions_before = {}
        self.insertions_after = {}
        self.deletions = 0
        self.was_deleted = False


class Sounds_Info:
    def __init__(self):
        #list of Sound class instances
        self.sound_information = []
    def get_sounds(self):
        return list(map(lambda x: x.value, self.sound_information))
    def add_sound(self, value, is_first):
        self.sound_information.append(Sound(value, is_first))
    def print_sounds(self):
        for sound in self.sound_information:
            print ('sound', sound.value)
            print ('occurrences', sound.occurences)
            print ('deletions', sound.deletions)
            print ('was first sound', sound.was_first_sound)
            print ('replacements', sound.replacings)
            print ('insertions before', sound.insertions_before)
            print ('insertions after', sound.insertions_after)
            
    def find_sound_index(self, sound):
        return self.get_sounds().index(sound)
    
    def get_key_with_max_values(self, dictionary):
        vals = list(dictionary.values())
        keys = list(dictionary.keys())
        return keys[vals.index(max(vals))]
    
    def find_last_occurrence(self, value, array):
        try:
            return max(loc for loc, val in enumerate(array) if val == value)
        except ValueError:
            return -1
    
    def check_sorted_descending_order(self, array):
        return all(array[i] >= array[i+1] for i in range(len(array) - 1))
    
    
    def update(self, correct_transcription, accented_transcription):
        # add new sounds
        for i, cur_sound in enumerate(correct_transcription):
            if cur_sound not in self.get_sounds():
                # if not in list of sounds
                if i: #not the first sound
                    self.add_sound(cur_sound, False)
                else: #is the first sound
                    self.add_sound(cur_sound, True)
            else:
                cur_sound_info = self.sound_information[self.find_sound_index(cur_sound)]
                if i: #if not the first sound
                    cur_sound_info.occurences += 1
                else: #if the first sound
                    cur_sound_info.occurences += 1
                    cur_sound_info.was_first_sound += 1
        #update information
        try: 
            s = difflib.SequenceMatcher(None, correct_transcription, accented_transcription)
        except NameError:
            try:
                import difflib
            except ImportError:
                !pip install difflib
            s = difflib.SequenceMatcher(None, correct_transcription, accented_transcription)
        prev_sound_index = -1
        print (correct_transcription, accented_transcription)
        for tag, i1, i2, j1, j2 in s.get_opcodes():
            print ("%7s a[%d:%d] (%s) b[%d:%d] (%s)" %
               (tag, i1, i2, correct_transcription[i1:i2], j1, j2, accented_transcription[j1:j2]))
            if tag == 'equal':
                prev_sound_index = i2 - 1 #index for sound before new tag
            elif tag == 'replace':
                if correct_transcription[i1:i2] in self.get_sounds():
                    cur_sound_info = self.sound_information[self.find_sound_index(correct_transcription[i1:i2])]
                    if accented_transcription[j1:j2] in cur_sound_info.replacings:
                        cur_sound_info.replacings[accented_transcription[j1:j2]] += 1
                    else: #if not in replacings
                        cur_sound_info.replacings[accented_transcription[j1:j2]] = 1
                else: #if current_transcription[i1:i2] not in self.get_sounds()
                    if i1: #if not the first sound
                        self.add_sound(correct_transcription[i1:i2], False)
                    else:
                        self.add_sound(correct_transcription[i1:i2], True)
                    cur_sound_info = self.sound_information[self.find_sound_index(correct_transcription[i1:i2])]
                    cur_sound_info.replacings[accented_transcription[j1:j2]] = 1
            elif tag == 'insert':
                prev_sound = correct_transcription[i1-1]
                if i2 < len(correct_transcription):
                    next_sound = correct_transcription[i2]
                else:
                    next_sound = None
                #update information for prev_sound
                cur_sound_info = self.sound_information[self.find_sound_index(prev_sound)]
                if accented_transcription[j1:j2] in cur_sound_info.insertions_after:
                    cur_sound_info.insertions_after[accented_transcription[j1:j2]] += 1
                else: #if not in replacings
                    cur_sound_info.insertions_after[accented_transcription[j1:j2]] = 1
                #update information for next sound
                if next_sound:
                    cur_sound_info = self.sound_information[self.find_sound_index(next_sound)]
                    if accented_transcription[j1:j2] in cur_sound_info.insertions_before:
                        cur_sound_info.insertions_before[accented_transcription[j1:j2]] += 1
                    else: #if not in replacings
                        cur_sound_info.insertions_before[accented_transcription[j1:j2]] = 1
            elif tag == 'delete':
                if correct_transcription[i1:i2] in self.get_sounds():
                    cur_sound_info = self.sound_information[self.find_sound_index(correct_transcription[i1:i2])]
                    cur_sound_info.deletions += 1
                    if not cur_sound_info.was_deleted:
                        cur_sound_info.was_deleted = True
                else:
                    if i1: #if not the first sound
                        self.add_sound(correct_transcription[i1:i2], False)
                        cur_sound_info = self.sound_information[self.find_sound_index(correct_transcription[i1:i2])]
                        cur_sound_info.deletions += 1
                    else:
                        self.add_sound(correct_transcription[i1:i2], True)
                    cur_sound_info = self.sound_information[self.find_sound_index(correct_transcription[i1:i2])]
                    cur_sound_info.deletions += 1
                    cur_sound_info.was_deleted = True
                    
    def fit(self, dataset):
        # dataset should be an array of pairs like [[correct_word, incorrect_word],
        #                                           [correct_word, incorrect_word], ...]
        for correct_word, incorrect_word in dataset:
            self.update(correct_word, incorrect_word)

    def replace_sound(self, correct_word, sound, index):
        initial_word = correct_word
        cur_sound_info = self.sound_information[self.find_sound_index(sound)]
        for replaced_sound in cur_sound_info.replacings.keys():
            initial_word = initial_word[:index] + replaced_sound + initial_word[index + 1:]
        return initial_word
    
    def generate_replacings(self, word):
        output = []
        output.append(word)
        for i, sound in enumerate(word):
            if sound in self.get_sounds():
                cur_sound_info = self.sound_information[self.find_sound_index(sound)]
                for re in cur_sound_info.replacings.keys():
                    new_word = word[:i] + re + word[i+1:]
                    if new_word not in output:
                        output.append(new_word)
        return output
    
    
    
    def generate_replacings_proba(self, word):
        output = []
        for i, sound in enumerate(word):
            if sound in self.get_sounds():
                cur_sound_info = self.sound_information[self.find_sound_index(sound)]
                for re, count in cur_sound_info.replacings.items():
                    new_word = word[:i] + re + word[i+1:]
                    proba = count / cur_sound_info.occurences
                    if new_word not in output:
                        output.append((new_word, proba))
        output.sort(key=lambda x: x[1], reverse=True)
        return output
    
    
    
    
    def generate_multiple_replacings(self, correct_word, max_ngram_size):
        output = []
        output.append(correct_word)
        for window_size in range(max_ngram_size):
                    for left, right in zip(range(len(correct_word) - window_size), \
                                            range(window_size + 1, len(correct_word) + window_size + 1)):
                        sound = correct_word[left:right]
                        if sound in self.get_sounds():
                            cur_sound_info = self.sound_information[self.find_sound_index(sound)]
                            for re in cur_sound_info.replacings.keys():
                                new_word = correct_word[:left] + re + correct_word[right:]
                                if new_word not in output:
                                    output.append(new_word)

        return output
    
    
    def generate_multiple_replacings_proba(self, correct_word, max_ngram_size):
        output = []
        #output.append(correct_word)
        for window_size in range(max_ngram_size):
                    for left, right in zip(range(len(correct_word) - window_size), \
                                            range(window_size + 1, len(correct_word) + window_size + 1)):
                        sound = correct_word[left:right]
                        if sound in self.get_sounds():
                            cur_sound_info = self.sound_information[self.find_sound_index(sound)]
                            for re, n_replacements in cur_sound_info.replacings.items():
                                new_word = correct_word[:left] + re + correct_word[right:]
                                proba = n_replacements / cur_sound_info.occurences
                                if new_word not in output:
                                    output.append((new_word, proba))
        output.sort(key=lambda x: x[1], reverse=True)

        return output
    
    
    
    
    
    def generate_insertions(self, word):
        output = []
        output.append(word)
        for i, sound in enumerate(word):
            if sound in self.get_sounds():
                cur_sound_info = self.sound_information[self.find_sound_index(sound)]
                for re in cur_sound_info.insertions_after.keys():
                    new_word = word[:i+1] + re + word[i+1:]
                    if new_word not in output:
                        output.append(new_word)
                    if i: #if not the first letter
                        for re in cur_sound_info.insertions_before.keys(): 
                            new_word = word[:i] + re + word[i:]
        return output
    
    
    
    def generate_insertions_proba(self, word):
        output = []
        #output.append(word)
        for i, sound in enumerate(word):
            if sound in self.get_sounds():
                cur_sound_info = self.sound_information[self.find_sound_index(sound)]
                for re, n_re in cur_sound_info.insertions_after.items():
                    new_word = word[:i+1] + re + word[i+1:]
                    proba = n_re / cur_sound_info.occurences
                    if new_word not in output:
                        output.append((new_word, proba))
                    if i: #if not the first letter
                        for re, n_re in cur_sound_info.insertions_before.items(): 
                            new_word = word[:i] + re + word[i:]
                            proba = n_re / cur_sound_info.occurences
                            #output.append((new_word, proba))
        output.sort(key=lambda x: x[1], reverse=True)
        return output
    
    
    
    
    
    def generate_deletions(self, word):
        output = []
        output.append(word)
        for i, sound in enumerate(word):
            if sound in self.get_sounds():
                cur_sound_info = self.sound_information[self.find_sound_index(sound)]
                if cur_sound_info.was_deleted:
                    new_word = word[:i] + word[i+1:]
                    output.append(new_word)
        return output
    
    
        
    def generate_deletions_proba(self, word):
        output = []
        #output.append(word)
        for i, sound in enumerate(word):
            if sound in self.get_sounds():
                cur_sound_info = self.sound_information[self.find_sound_index(sound)]
                if cur_sound_info.deletions > 0:
                    new_word = word[:i] + word[i+1:]
                    proba = cur_sound_info.deletions / cur_sound_info.occurences
                    output.append((new_word, proba))
        output.sort(key=lambda x: x[1], reverse=True)
        return output
    
    
    
    def generate(self, word, ngram_size=3):
        deletions = self.generate_deletions(word)
        insertions = self.generate_insertions(word)
        replacings = self.generate_multiple_replacings(word, ngram_size)
        ans = {'Replacings': replacings, 'Insertions': insertions, 'deletions': deletions}
        for key, value in ans.items():
            print (key, ":")
            for item in value:
                print (item, sep=' ', end=', ', flush=True)
            print ("\n _______________________ \n")
        return ans
    
    def generate_accented_samples(self, word):
        print ('Original word: ', word)
        print ('\n ---------------------- \n')
        transcription = convert_without_stress(word)
        #print ('\n ---------------------- \n')
        print ('Phonetic transcription: ', transcription)
        print ('\n ---------------------- \n')
        return self.generate(transcription)
    
    def generate_top(self, word, ngram_size=3, 
                     n_deletions=1, n_insertions=1, n_replacements=1):
        deletions = self.generate_deletions_proba(word)
        assert self.check_sorted_descending_order([x[1] for x in deletions])
        insertions = self.generate_insertions_proba(word)
        assert self.check_sorted_descending_order([x[1] for x in insertions])
        replacements = self.generate_multiple_replacings_proba(word, ngram_size)
        assert self.check_sorted_descending_order([x[1] for x in replacements])
        top_deletions = deletions[:self.find_last_occurrence(1, [x[1] for x in deletions]) + 1 + n_deletions]
        top_insertions = insertions[:self.find_last_occurrence(1, [x[1] for x in insertions]) + 1 + n_insertions]
        #print (self.find_last_occurrence(1, [x[1] for x in replacements]), 1 + n_replacements)
        top_replacements = replacements[:
                                        self.find_last_occurrence(1, [x[1] for x in replacements]) 
                                        + 1 + n_replacements]
        #ans = {'replacements': [x[0] for x in top_replacements], 
        #       'insertions': [x[0] for x in top_insertions], 'deletions': [x[0] for x in top_deletions]}
        ans = {'replacements': top_replacements, 
               'insertions': top_insertions, 'deletions': top_deletions}
        ans_list = []
        for x in ans.values():
            ans_list = ans_list + x
        
        for key, value in ans.items():
            print (key, ":")
            for item in value:
                print (item, sep=' ', end=', ', flush=True)
            print ("\n _______________________ \n")
        return ans, ans_list





#Sample use case

test_info = Sounds_Info()
input_array = np.load(<your file>)

'''
input_array is an array with pairs:

    [['pliz', 'pliiz'],
       ['pliz', 'plez'],
       ['pliz', 'pləiz'], ... ]
'''

test_info.fit(input_array)

test_word = test_info.generate_accented_samples('milk')

'''
Output:

Original word:  milk

 ---------------------- 

Phonetic transcription:  mɪlk

 ---------------------- 

Replacings :
mɪlk, nɪlk, mʊlk, mulk, mɛlk, miəlk, məlk, milk, melk, mɪərk, mɪŋk, mɪrk, mɪwk, mɪlg, mɪlh, 
 _______________________ 

Insertions :
mɪlk, məɪlk, muɪlk, mɪllk, mɪɛlk, mɪðlk, mɪəlk, mɪslk, mɪzlk, mɪflk, mɪθlk, mɪlək, mɪltk, mɪluk, mɪlɛk, mɪlɪk, mɪlkj, mɪlkɛ, mɪlkə, mɪlks, mɪlki, mɪlkɪ, 
 _______________________ 

deletions :
mɪlk, mlk, mɪk, mɪl, 
 _______________________ 


'''


