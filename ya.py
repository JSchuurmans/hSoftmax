import os
import xml.etree.ElementTree
import xmltodict
# list with tuples (sentence, tag)

datapath = os.path.join('data','ya','Yahoo','Yahoo.ESA')

tags = os.listdir(datapath)
print(tags)

data = []
for tag in tags:
# tag = tags[0]
    file_path = os.path.join(datapath,tag)
    with open(file_path) as f:
        for line in f:
            if line[0] =='<' or skip_next:
                skip_next = False
                continue
            else:
                skip_next = True
                line.strip().replace('</TEXT>', '').replace('\n', ' ')
            data.append((line,tag))

# print(data[0])

# print(data[1])

# print(data[2])

    # xml = f.read()
    # print(type(xml))
    # print(xml[:10])
    # xmltodict.parse(f'<r>{xml}</r>')

