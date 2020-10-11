import sys
import re
import requests
import bs4
import json

"""Create Classification Data of Extended Named Entities"""


def request():

    print('Requesting to ene site...')
    # url = 'http://liat-aip.sakura.ne.jp/ene/ene8/definition_jp/html/enedetail.html'
    url = 'http://liat-aip.sakura.ne.jp/ene/ene8/definition_jp/html/js/data/enedetail_data_0409.js'
    r = requests.get(url)
    r.raise_for_status()

    return r.text


def parse(html_text):
    soup = bs4.BeautifulSoup(html_text, 'html.parser')
    ene_tags = soup.select('div.ene_title')

    records = []
    for ene_tag in ene_tags:
        children = [child for child in ene_tag.children]
        assert len(children) == 1, 'Unexpected structure.'
        num, class_name, label = ene_tag.text.split()
        label = label[1:-1]
        records.append([num, class_name, label])
    return records


def add_mask(records):
    new_records = []
    for r in records:
        num, class_name, label = r

        num_list = num.split('.')
        if num_list[0] == '1':
            if len(num_list) == 1:
                mask = 'IDENTITYMASK'
            else:
                if num_list[1] in ['0', '1', '2', '3', '4']:
                    mask = 'IDENTITYMASK'
                elif num_list[1] in ['5', '6']:
                    mask = 'PLACEMASK'
                elif num_list[1] in ['7', '8', '9', '10', '11', '12']:
                    mask = 'THINGMASK'
                else:
                    raise ValueError('Unexpected type: %s, %s, %s' %
                                     (num, class_name, label))
        elif num_list[0] == '2':
            mask = 'TEMPORALMASK'
        elif num_list[0] == '3':
            mask = 'NUMERICMASK'
        elif num_list[0] == '0' or num_list[0] == '9':
            mask = 'THINGMASK'
        else:
            raise ValueError('Unexpected type: %s, %s, %s' %
                             (num, class_name, label))

        new_records.append([num, class_name, label, mask])
    return new_records


def to_csv(records, csv_path):
    lines = [','.join(r) for r in records]
    data = '\n'.join(lines)

    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write(data)


def to_json(records, json_path):
    label2mask = {}
    for r in records:
        _, _, label, mask = r
        label2mask[label] = mask

    with open(json_path, 'w') as f:
        json.dump(label2mask, f)


def extract_extended_named_entities(prefix_path):
    text = request()
    html_text = re.search('\\/\\*(.+)\\*\\/', text, flags=re.DOTALL).group(1)
    records = parse(html_text)
    records = add_mask(records)

    csv_path = prefix_path + '.csv'
    json_path = prefix_path + '.json'
    to_csv(records, csv_path)
    to_json(records, json_path)
    print('Created Classification Data of Extended Named Entities: %s' % csv_path)
    print('Created Classification Data of Extended Named Entities: %s' % json_path)


if __name__ == '__main__':
    prefix_path = sys.argv[1]

    extract_extended_named_entities(prefix_path)
