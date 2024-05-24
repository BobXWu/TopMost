import os
from sklearn.datasets import fetch_20newsgroups
from topmost.data import file_utils


def download_save(output_dir):
    names = ['talk.religion.misc', 'comp.windows.x', 'rec.sport.baseball', 'talk.politics.mideast', 'comp.sys.mac.hardware', 'sci.space', 'talk.politics.guns', 'comp.graphics', 'comp.os.ms-windows.misc', 'soc.religion.christian', 'talk.politics.misc', 'rec.motorcycles', 'comp.sys.ibm.pc.hardware', 'rec.sport.hockey', 'misc.forsale', 'sci.crypt', 'rec.autos', 'sci.med', 'sci.electronics', 'alt.atheism']

    # prefixes = set([name.split('.')[0] for name in names])

    groups = {'20ng_all': names}
    # for prefix in prefixes:
    #     elements = [name for name in names if name.split('.')[0] == prefix]
    #     groups['20ng_' + prefix] = elements

    print(groups)

    for group in groups.keys():
        if len(groups[group]) > 1:
            for subset in ['train', 'test', 'all']:
                download_articles(group, groups[group], subset, output_dir)


def download_articles(name, categories, subset, output_dir):
    print("name: ", name)
    print("categories: ", categories)
    print("subset: ", subset)

    data = []
    print("Downloading articles")
    newsgroups_data = fetch_20newsgroups(subset=subset, categories=categories, remove=())

    for i in range(len(newsgroups_data['data'])):
        line = newsgroups_data['data'][i]
        data.append({'text': line, 'group': newsgroups_data['target_names'][newsgroups_data['target'][i]]})

    print("data size: ", len(data))
    print("Saving to", output_dir)
    file_utils.make_dir(output_dir)
    file_utils.save_jsonlist(data, f"{output_dir}/{subset}.jsonlist")
