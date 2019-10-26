import os
import tarfile
from xml.dom import minidom
import pickle
import re

"""
TO DOWNLOAD DATASET RUN: 


TO UNZIP DOWNLOADED DATASET RUN: 
unzip(source = 'images.tgz', destination = 'chest_images')
unzip(source = 'reports.tgz', destination = 'chest_reports')

TP RUN PREPROCESSING ON THE DATASET RUN:
TAG_VOCAB, TAGS_DISTRIBUTION = process_all_reports('chest_images', 'chest_reports/ecgen-radiology')

this will result in having ....
"""


# --------------- UNZIP DOWNLOADED DATA ---------------------
def unzip(path_from, path_to):
    """
    Unzip .tgz files to directory
    :param path_from: path to file that you want to unzip
    :param path_to: path to directory where you want to extract to
    :return:
    """
    if not os.path.exists(path_to):
        os.makedirs(path_to)
    tar = tarfile.open(path_from)
    tar.extractall(path=path_to)
    tar.close()


# ---------------- PREPROCESS REPORTS --------------------------
# ---------------- CREATE IMAGE - REPORT MAPPING --------------------
def remove_extra_charachters(report, tags):
    """
    Convert to lower case and remove non-alpha characters.
    :param report: string of words, symbols and numbers
    :param tags: list of tags, each tag is a string of one word or multiple words
    :return: report in lower case with only letters and dots, each sentence is separated buy dot.
    :return: tags converted to lowercase
    """
    tags = [t.lower() for t in tags]
    reg = re.compile('[^a-zA-Z.? ]')
    report = reg.sub(' ', report)
    report = report.lower()
    report = re.sub('xx+', '', report)
    report = re.sub('  +', ' ', report)
    report.replace('?', '.')
    report = report.lower()
    return report, tags


def process_report(xml_report_path):
    """
    Get useful information from xml report  e.g. IMPRESSION FINDINGS TAGS and images associated with that report
    :param xml_report_path: path to report, report is in xml format
    :return:
    """
    report, tags = '', []

    captions = ['IMPRESSION', 'FINDINGS']
    xml_report = minidom.parse(xml_report_path)
    AbstractText = xml_report.getElementsByTagName('AbstractText')
    for elem in AbstractText:
        if elem.attributes['Label'].value in captions:
            try:
                report += elem.firstChild.data
            except AttributeError:
                continue

    MIT = xml_report.getElementsByTagName('automatic')
    tags = set([m.firstChild.data for m in MIT])
    MeSH = xml_report.getElementsByTagName('major')
    for m in MeSH:
        tag_data = m.firstChild.data.split('/')
        [tags.add(td) for td in tag_data]

    images_id = [img.attributes['id'].value for img in xml_report.getElementsByTagName('parentImage')]
    report, tags = remove_extra_charachters(report, tags)
    report = report.split('.')
    report_processed = []
    for r in range(len(report)):
        sentence = [word for word in r.split(' ') if len(word) > 0]
        if len(sentence) > 0:
            report_processed.append(sentence)

    return report_processed, tags, images_id


def process_all_reports(reports_dir):
    """
    Applies processing to each report in the reports directory.
    :param reports_dir: path to directory containing reports in xml format
    :return: IMG_REPORT: mapping between Image name (not path to image) and corresponding processed report, this is
            saved to IMG_REPORT.pickle file
    :return: IMG_TAG: mapping between Image name (not path to image) and corresponding list of tags, also saved into
            IMG_TAG.pickle
    :return: TAG_VOCAB: a set of all possible tags
    :return: WORD_VOCAB: a set of all possible words occurring in the reports
    """
    reports_paths = os.listdir(reports_dir)

    IMG_REPORT, IMG_TAG = {}, {}
    TAG_VOCAB, WORD_VOCAB = set(), set()
    IMG_NORMAL_ABNORMAL = {}

    for r_path in reports_paths:
        report, tags, images_id = process_report(f'{reports_dir}/{r_path}')
        for img in images_id:
            IMG_REPORT[img] = report
            IMG_TAG[img] = tags
            if len(tags) == 1 and tags[0] == 'normal':
                IMG_NORMAL_ABNORMAL[img] = 1
            else:
                IMG_NORMAL_ABNORMAL[img] = 0

        [TAG_VOCAB.add(t) for t in tags]
        [WORD_VOCAB.add(w) for sentence in report for w in sentence]

    with open('IMG_NORMAL_ABNORMAL.pickle', 'wb') as f:
        pickle.dump(IMG_NORMAL_ABNORMAL, f)

    with open('IMG_REPORT.pickle', 'wb') as f:
        pickle.dump(IMG_REPORT, f)

    with open('IMG_TAG.pickle', 'wb') as f:
        pickle.dump(IMG_TAG, f)

    return IMG_REPORT, IMG_TAG, TAG_VOCAB, WORD_VOCAB
