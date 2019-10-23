import os
import tarfile
from xml.dom import minidom
import pickle
import re


"""
TO UNZIP DOWNLOADED DATASET RUN: 
unzip(source = 'images.tgz', destination = 'chest_images')
unzip(source = 'reports.tgz', destination = 'chest_reports')

TP RUN PREPROCESSING ON THE DATASET RUN:
TAG_VOCAB, TAGS_DISTRIBUTION = process_all_reports('chest_images', 'chest_reports/ecgen-radiology')

this will result in having ....
"""

#--------------- UNZIP DOWNLOADED DATA ---------------------
def unzip(path_from, path_to):
  """
  path_from: path to file that you want to unzip
  path_to: path to directory where you want to extract to
  """
  if not os.path.exists(path_to):
    os.makedirs(path_to)
  tar = tarfile.open(path_from)
  tar.extractall(path = path_to)
  tar.close()

# ---------------- PREPROCESS REPORTS --------------------------
# ---------------- CREATE IMAGE - REPORT MAPPING --------------------
def remove_extra_charachters(report, tags):
    """
    convert to lower case and remove non-alpha characters
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
    get useful information from xml report
    e.g. IMPRESSION FINDINGS TAGS and images assosiated with report
    return: report_id: id of the report, images accosiate
    return: report
    return: tags
    """
    report, tags = '', []

    captions = 'IMPRESSION' 'FINDINGS'
    xml_report = minidom.parse(xml_report_path)
    AbstractText = xml_report.getElementsByTagName('AbstractText')

    for elem in AbstractText:
        if elem.attributes['Label'].value in captions:
            try:
                report += elem.firstChild.data
            except AttributeError:
                continue

    MIT = xml_report.getElementsByTagName('automatic')
    tags = [m.firstChild.data for m in MIT]
    if len(tags) == 0:
        MeSH = xml_report.getElementsByTagName('major')
        tags = []
        for m in MeSH:
            tag_data = m.firstChild.data.split('/')
            for td in tag_data:
                tags.append(td)
    images_id = [img.attributes['id'].value for img in xml_report.getElementsByTagName('parentImage')]
    tags = set(tags)
    report, tags = remove_extra_charachters(report, tags)
    report = report.split('.')
    report_copy = []
    for i in range(len(report)):
        s = report[i].split(' ')
        l = []
        for w in s:
            if len(w)>0:
                l.append(w)
        if len(l)>0:
            report_copy.append(l)
    return report_copy, tags, images_id


def process_all_reports(reports_dir):
    """
    imgs_path: path to directory containing images
    reports_path: path to directory containing reports in xml format
    """
    reports_paths = os.listdir(reports_dir)

    IMG_REPORT, IMG_TAG = {}, {}
    TAG_VOCAB, WORD_VOCAB = set(), set()
    IMG_NORMAL_ABNORMAL = {}


    TAGS_DISTRIBUTION = []

    for r_path in reports_paths:
        report, tags, images_id = process_report(f'{reports_dir}/{r_path}')
        for img in images_id:
            IMG_REPORT[img] = report
            IMG_TAG[img] = tags
            if len(tags)==1 and tags[0] =='normal':
                IMG_NORMAL_ABNORMAL[img] = 1
            else: IMG_NORMAL_ABNORMAL[img] = 0

        for t in tags:
            TAG_VOCAB.add(t)
            TAGS_DISTRIBUTION.append(t)
        for s in report:
            for w in s:
                WORD_VOCAB.add(w)

    with open('IMG_NORMAL_ABNORMAL.pickle', 'wb') as f:
        pickle.dump(IMG_NORMAL_ABNORMAL, f)

    with open('IMG_REPORT.pickle', 'wb') as f:
        pickle.dump(IMG_REPORT, f)

    with open('IMG_TAG.pickle', 'wb') as f:
        pickle.dump(IMG_TAG, f)

    return TAG_VOCAB, TAGS_DISTRIBUTION