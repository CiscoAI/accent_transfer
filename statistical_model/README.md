statistical_model.py

Class Sound contains imporant information that being updated during training.

value - string. Represents sound
occurences - int. Total number of the sound occurrences in the dataset. (We made spelling mistake in the very beginning and now it's too many things to change)
was_first_sound - int. Total number of the sound occurrences as a first letter of particular word in the dataset.
replacings - dict. Replacements providing the number of times each sound replaced the given one.
insertions_before, insertions_after - dict. Insertions  providing  the  number  of  timeseach  sound  was  inserted  before  or  after  thegiven one.
deletions - int. Deletions providing the number of times thegiven sound was deleted from the GAE pro-nunciation.
was_deleted - bool. Deprecated


Class Sounds_Info stores information obtained from the dataset
