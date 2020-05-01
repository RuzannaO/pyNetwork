from collections import defaultdict
def nonempty(a):
    return a
def parse_ini(path):
    # reads txt file data, the list "lines' is comprised of lines from the txt file
    with open(path) as rdr:
        lines = rdr.read().splitlines()
    # clear empty elements, i.e. empty lines
    new_lines = list(filter(nonempty, lines))
    # defines indexes which correspond to sections and which to the content data (lists - sections_index and content_index)
    sections_index = []
    content_index = []
    for i in range(0, len(new_lines)):
        if new_lines[i][0] == "[" and new_lines[i][-1] == "]":
            sections_index.append(i)
        else:
            content_index.append(i)

    # cleans [] brackets from sections' names
    for i in sections_index:
        new_lines[i] = new_lines[i].replace("[", "")
        new_lines[i] = new_lines[i].replace("]", "")

    # defines the final dictionary
    dict2=defaultdict(set,{k:[] for k in sections_index})

    dict1 = defaultdict(set)
    for k in range(0, len(sections_index)):

        for j in content_index:
            if k == len(sections_index) - 1 and j > sections_index[k]:
                dict1[new_lines[sections_index[k]]].add(new_lines[j])

            else:
                if j > sections_index[k] and j < sections_index[k + 1]:
                    dict1[new_lines[sections_index[k]]].add(new_lines[j])
            dict1[new_lines[sections_index[-1]]].add("")
            dict1[new_lines[sections_index[-1]]].remove("")
    return (dict(dict1))


print(parse_ini("test.ini"))
