# coding: utf-8
'''
tensorflow: 2.9.0
tensorflow-datasets: 2.1.0
python: 3.7
'''
import os
import tensorflow as tf
from tensorboard.plugins import projector
import numpy as np
import configparser
config = configparser.ConfigParser()
config.read('conf/only_pos_config.ini')
# gpu = config["model"]["train_gpu"]
from db import Mongo

TOP_TAGS = ['1042015:newTagCategory_026',
    '1042015:newTagCategory_006',
    '1042015:newTagCategory_012',
    '1042015:newTagCategory_047',
    '1042015:newTagCategory_046',
    '1042015:newTagCategory_041',
    '1042015:newTagCategory_011',
    '1042015:newTagCategory_050',
    '1042015:newTagCategory_007',
    '1042015:newTagCategory_009',
    '1042015:newTagCategory_016',
    '1042015:newTagCategory_030',
    '1042015:newTagCategory_042',
    '1042015:newTagCategory_045',
    '1042015:newTagCategory_036']

def get_item2tag():
    mongo_item = Mongo(user=config['db']['user'],
                  password=config['db']['password'],
                  host=config['db']['host'],
                  database=config['db']['database'],
                  collections="oasis_item")

    data_item = mongo_item.find({}, keys=['mid', '10540'])
    item2tag = {}
    for _data in data_item:
        mid = _data["mid"]
        # tag = _data["10540"] if '10540' in _data else 'unk'
        if '10540' in _data:
            tag = _data['10540']
            tag = tag.strip()
            if tag:
                if '|' in tag:
                    tag = _data['10540'].split('|')[0]
                else:
                    tag = _data['10540']

                tag = tag.split('@')[0]
            else:
                tag = 'unk'
        else:
            tag = 'unk'

        item2tag[mid] = tag

    return item2tag

item2tag = get_item2tag()
print('item2tag done.')

# print(item2tag)
def embed_vis(log_dir, mid_list, embeddings):

    # Save Labels separately on a line-by-line manner.
    with open(os.path.join(log_dir, 'metadata.tsv'), "w") as f:
        f.write("mid\ttag\n")
        for mid in mid_list:
            tag = item2tag[mid] if mid in item2tag else ''
            f.write(mid+"\t"+ tag + "\n")

    # Save the weights we want to analyze as a variable. Note that the first
    # value represents any unknown word, which is not in the metadata, here
    # we will remove this value.
    # name is Variable:0
    weights = tf.Variable(embeddings)
    # Create a checkpoint from embedding, the filename and key are the
    # name of the tensor.
    checkpoint = tf.train.Checkpoint(embedding=weights)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))

    # Set up config.
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    # The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`.
    embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"

    # embedding.tensor_name = "aaa"
    embedding.metadata_path = 'metadata.tsv'
    summary_writer = tf.summary.FileWriter(log_dir)

    projector.visualize_embeddings(summary_writer, config)

def embed_vis_from_file(embed_path="./item_embedding_mingxing", vis_path='./checkpoint/mingxing', top_filter=False):
    '''

    :param embed_path:
    :type embed_path:
    :param vis_path:
    :type vis_path:
    :param top_filter: 是否过滤。根据mid的tag
    :type top_filter:
    :return:
    :rtype:
    '''
    with open(embed_path, "r") as f:
        line = f.readline()
        embeddings = []
        mid_list = []
        while line:
            line = line.strip()
            arr = line.split(':')
            mid = arr[0]

            if top_filter:
                if mid in item2tag and item2tag[mid] not in TOP_TAGS:
                    line = f.readline()
                    continue

            mid_list.append(mid)

            emb = arr[1]
            emb_arr = emb.split(',')
            emb_float_arr = [float(ele)for ele in emb_arr]
            embeddings.append(emb_float_arr)

            line = f.readline()
    print('mid_list and embeddings ready.')

    assert len(mid_list) == len(embeddings)

    embed_vis(vis_path, mid_list, np.array(embeddings))

if __name__ == '__main__':
    # qingxin
    embed_vis_from_file(embed_path='./embedding/item_embedding', vis_path='./checkpoint/qingxin')
    print('qingxin done')

    embed_vis_from_file(embed_path='./embedding/item_embedding', vis_path='./checkpoint/qingxin_large', top_filter=True)
    print('qingxin large done')

    # wode
    # main(embed_path='./embedding/item_embedding_mingxing', vis_path='./checkpoint/mingxing')
    # print('mingxing done')

    embed_vis_from_file(embed_path='./embedding/item_embedding_mingxing', vis_path='./checkpoint/mingxing_large', top_filter=True)
    print('mingxing large done')