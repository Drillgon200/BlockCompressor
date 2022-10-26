# Screw this I'm not doing manual data entry

import sys

lineIndex = 0
numberProcessed = 0
partitionData = open(sys.argv[1]);
lines = [l.strip() for l in filter(lambda line: len(line.strip()) > 0, partitionData.readlines())]

def process_row():
    global lineIndex
    global numberProcessed
    global lines
    lineIndex += 8
    for col in range(8):
        output = "{ //" + str(numberProcessed) + "\n"
        for y in range(4):
            idx = lineIndex + y * 32 + col * 4
            output += lines[idx]
            for x in range(1, 4):
                output += ", " + lines[idx + x] + ("," if y < 3 and x == 3 else "")
            output += "\n"
        numberProcessed += 1
        output += "}," if numberProcessed < 64 else "}"
        print output
    lineIndex += 16 * 8

for i in range(8):
    process_row()

partitionData.close()
