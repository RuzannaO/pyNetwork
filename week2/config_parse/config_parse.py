from collections import defaultdict

def parser_ini(path):

    # function creates a dictionary from a section content line
    def newdict(a):
        newdict = {}
        aaa = str(a).split(",")
        for i in range(0, len(aaa)):
            if aaa[i][aaa[i].index("=") + 1:].isnumeric():
                newdict[aaa[i][:aaa[i].index("=")]] = int(aaa[i][aaa[i].index("=") + 1:])
            else:
                newdict[aaa[i][:aaa[i].index("=")]] = aaa[i][aaa[i].index("=") + 1:]
        return newdict


    def nonempty(a):
        return a

    # reads txt file data, creates a list "txtlines' comprised of lines from the txt file
    with open(path) as rdr:
        txtlines = rdr.read().splitlines()
    # removes empty elements, i.e. empty lines
    lines = list(filter(nonempty, txtlines))
    # defines indexes which correspond to the sections and which to the content data (resulting lists - sections_index and content_index)
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

    # defines the result dictionary structure

    dict1 = defaultdict(set)
    for k in range(0, len(sections_index)):

        for j in content_index:
            if k == len(sections_index) - 1 and j > sections_index[k]:
                dict1[lines[sections_index[k]]].add(j)

            else:
                if j > sections_index[k] and j < sections_index[k + 1]:
                    dict1[lines[sections_index[k]]].add(j)
                else:
                    dict1[lines[sections_index[k]]].add("")
                    dict1[lines[sections_index[k]]].remove("")

    # the resulting dictionary
    dict2 = defaultdict(dict, {k:{} for k in dict1.keys()})
    for k in dict1.keys():
        for j in dict1[k]:
            dict2[k][list(newdict(lines[j]))[0]] = newdict(lines[j])[str(list(newdict(lines[j]))[0])]

    return dict(dict2)



print(parser_ini("test.ini"))



