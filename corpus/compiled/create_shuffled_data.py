from random import shuffle
filenames = ['group1_nature.tsv',
             'group2_dateandtime.tsv',
             'group3_officework.tsv',
             'group4_health.tsv',
             'group5_food.tsv',
             'group6_shopping.tsv',
             'group7_travel.tsv',
             'group9_school.tsv',
             'tgl.txt']

sentences = []

for fname in filenames:
    file = open(fname, 'r')
    for line in file:
        if line[-1] != '\n':
            line = line + '\n'
        sentences.append(line)
    file.close()

for i in range(20):
    shuffle(sentences)

output_fname = 'new_shuffled.tsv'
output_file = open(output_fname, 'w')
for line in sentences:
    output_file.write(line)
output_file.close()
