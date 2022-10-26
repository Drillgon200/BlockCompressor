# Screw this I'm not doing manual data entry

import sys

lineIndex = 0
partitionData = open(sys.argv[1]);
lines = [l.strip() for l in filter(lambda line: len(line.strip()) > 0, partitionData.readlines())]

data = ""
for i in range(8):
    lineIndex += 8
    for j in range(8):
        data += lines[lineIndex] + (", " if i*8 + j < 63 else "")
        lineIndex += 1
print(data)

partitionData.close();
