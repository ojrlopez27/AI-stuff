import json


def save_txt(filename, lines, folder='', mode='wt'):
    text_file = open(folder + filename, mode)
    if type(lines) == list:
        text_file.writelines(lines)
    else:
        text_file.write(lines)
    text_file.close()


def load_txt(filename, folder=''):
    with open(folder + filename, 'rt') as of:
        text_file = of.readlines()
    text = ' '.join(text_file)
    text = text.replace('\n', ' ').replace('  ', ' ')
    return text


def load(filename, folder='', default=None):
    try:
        path = folder + filename
        with open(path) as data:
            return json.load(data)
    except Exception as e:
        print(e)
        return default


def save(filename, conversation, folder='', indent='\t', sort_keys=False):
    path = folder + filename
    with open(path, 'w') as outfile:
        json.dump(conversation, outfile, indent=indent, sort_keys=sort_keys)


def from_json(string):
    return json.loads(string)


def to_json(element):
    return json.dumps(element)