from collections import defaultdict


def nonempty(a):
    return a


def parse_ini(path):
    # reads txt file data, creates a list "txtlines' comprised of lines from the txt file
    with open(path) as rdr:
        txtlines = rdr.read().splitlines()
    # clear empty elements, i.e. empty lines
    lines = list(filter(nonempty, txtlines))
    # defines indexes which correspond to the sections and which to the content data (lists - sections_index and content_index)
    sections_index = []
    content_index = []
    for i in range(0, len(lines)):
        if lines[i][0] == "[" and lines[i][-1] == "]":
            sections_index.append(i)
        else:
            content_index.append(i)
    # cleans [] first and last brackets from sections' names
    for i in sections_index:
        lines[i] = lines[i][1:len(lines[i])-1]

    # validation  - section names do not contain extra ] [ brackets

    for i in sections_index:
        counter = 0
        if "[" in lines[i] or "]" in lines[i]:
            counter += 1
            print(f' ERROR! BAD SECTION NAME "{lines[i]}"')

    # validation - "=" available or not
    s = 0
    for j in content_index:
        if "=" not in lines[j]:
            print(f' ERROR! INACCURATE DATA ... MISSING "=" in "{lines[j]}" ')
            s += 1
    if s > 0 or counter > 0:
        return ""

    # defines the final dictionary
    dict2 = defaultdict(set, {k:[] for k in sections_index})

    dict1 = defaultdict(set)
    for k in range(0, len(sections_index)):

        for j in content_index:
            if k == len(sections_index) - 1 and j > sections_index[k]:
                dict1[lines[sections_index[k]]].add(lines[j])

            else:
                if j > sections_index[k] and j < sections_index[k + 1]:
                    dict1[lines[sections_index[k]]].add(lines[j])
                else:
                    dict1[lines[sections_index[k]]].add("")
                    dict1[lines[sections_index[k]]].remove("")

    return (dict(dict1))


print(parse_ini("test.ini"))
