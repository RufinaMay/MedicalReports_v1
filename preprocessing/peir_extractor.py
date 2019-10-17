# This file extracts images and report from PEIR Digital Library.
# http://peir.path.uab.edu/library/
# Each Image corresponds to one report

from requests_html import HTMLSession
from bs4 import BeautifulSoup
from urllib.request import urlopen, urlretrieve
import os

"""
TO EXTRACT ALL DATA FROM PEIR RUN: 
extract_all_images_descriptions(extract_categories_set())

TO EXTRACT ONLY GROSS CATEGORY:
gross = extract_gross_categories()
image_links = extract_image_links(gross)
"""

#------------------------Extract all data---------------------------------------
def extract_image(pic_link, i, img_dir='PEIR_images', report_dir='PEIR_reports'):
    """
    extract image and description from a given image link
    e.g. http://peir.path.uab.edu/library/picture.php?/17046/category/108
    """
    session = HTMLSession()

    # get picture from the pic link and description
    r = session.get(pic_link)
    about = r.html.find("p")
    description = about[1].text

    if 'GROSS' in description:
        html = urlopen(pic_link)
        soup = BeautifulSoup(html)
        imgs = soup.find_all('img')
        pic_url = 'http://peir.path.uab.edu/library' + imgs[0].get('src')[1:]

        pic_file = f'{img_dir}/img{i}.jpg'
        des_file = f'{report_dir}/des{i}.txt'
        urlretrieve(pic_url, pic_file)

        with open(des_file, 'w') as file:
            file.write(description)


def extract_all_images(categorie_link, i, img_dir='PEIR_images', report_dir='PEIR_reports'):
    """
    extract all images and descriptions from a link
    containing set of images,
    e.g. http://peir.path.uab.edu/library/index.php?/category/108
    """
    session = HTMLSession()
    r = session.get(categorie_link)
    # for each link on a page, if it contains a picture
    for pic_link in r.html.absolute_links:
        if 'picture' in pic_link:
            # get picture from the pic link and description
            extract_image(pic_link, i, img_dir, report_dir)
            i += 1

    return i


def extract_all_images_descriptions(categories):
    """
    вытаскивает картинки для всех категорий
    на вход просто категории внутри которых уже картинки и ничего другого
    """
    session = HTMLSession()
    path_save_imgs = 'PEIR_images'
    path_save_reports = 'PEIR_reports'

    if not os.path.exists(path_save_imgs):
        os.makedirs(path_save_imgs)
    if not os.path.exists(path_save_reports):
        os.makedirs(path_save_reports)

    # now extract all images and their descroptions
    i = 0
    for cat in categories:
        i = extract_all_images(cat, i)


def extract_categories_set():
    cats = ['http://peir.path.uab.edu/library/index.php?/category/2']
    cats.append('http://peir.path.uab.edu/library/index.php?/category/127')
    cats.append('http://peir.path.uab.edu/library/index.php?/category/106')
    double_hierarchy = [True, True, False]
    begin = 'http://peir.path.uab.edu/library/index.php?/category/'

    all_links_set = set()
    session = HTMLSession()

    for i in range(2):
        cat, h = cats[i], double_hierarchy[i]
        r = session.get(cat)
        links_set = set()
        for link in r.html.absolute_links:
            if link not in cats and link.startswith(begin) and not 'monthly' in link:
                links_set.add(link)

        if h:
            for link in links_set:
                subsub = session.get(link).html.absolute_links
                for l in subsub:
                    if l not in links_set and l.startswith(begin) and not 'monthly' in l:
                        all_links_set.add(l)
        else:
            all_links_set = all_links_set.union(links_set)

    return all_links_set

#----------------------- Extract only Gross category ------------------
def extract_gross_categories():
    """
    e.g. output: http://peir.path.uab.edu/library/index.php?/tags/4-gross/start-11500
    """
    url = 'http://peir.path.uab.edu/library/index.php?/tags/4-gross'
    session = HTMLSession()

    next_urls = {url}
    step = 100
    while step < 12901:
        next_urls.add(f'http://peir.path.uab.edu/library/index.php?/tags/4-gross/start-{step}')
        step += 100

    return next_urls


def extract_image_links(gross):
    """
    на вход данные в таком виде
    http://peir.path.uab.edu/library/index.php?/tags/4-gross/start-11500
    """
    session = HTMLSession()
    path_save_imgs = 'PEIR_images'
    path_save_reports = 'PEIR_reports'
    end = 'tags/4-gross'
    if not os.path.exists(path_save_imgs):
        os.makedirs(path_save_imgs)
    if not os.path.exists(path_save_reports):
        os.makedirs(path_save_reports)

    set_links = set()

    for cat in gross:
        links = session.get(cat).html.absolute_links
        for link in links:
            if 'picture' in link and link.endswith(end):
                set_links.add(link)

    return set_links


def save_image_report(pic_link, i, img_dir='PEIR_images', report_dir='PEIR_reports'):
    session = HTMLSession()

    # ink and description
    r = session.get(pic_link)
    about = r.html.find("p")
    description = about[1].text

    html = urlopen(pic_link)
    soup = BeautifulSoup(html)
    imgs = soup.find_all('img')
    pic_url = 'http://peir.path.uab.edu/library/' + imgs[0].get('src')

    pic_file = f'{img_dir}/img{i}.jpg'
    des_file = f'{report_dir}/des{i}.txt'
    urlretrieve(pic_url, pic_file)

    with open(des_file, 'w') as file:
        file.write(description)

