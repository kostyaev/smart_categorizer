import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
import tqdm
import glob
from sklearn.cross_validation import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import os
import shutil
import argparse
import sys


parser = argparse.ArgumentParser(description='Trainable categorization tool')
parser.add_argument(
        '--train_dir',
        type=str,
        help='Directory containing positive and optionally negative images to train on',
        required=True
)
parser.add_argument(
        '--target_data',
        type=str,
        help='Path to dir containing uncategorized data',
        required=True
)

parser.add_argument(
        '--result_dir',
        type=str,
        help='Path to save categorized data',
        required=True
)


args = parser.parse_args()


def chunks(l, n):
    return [l[i:i + n] for i in xrange(0, len(l), n)]


def get_rand_subset(data, n):
    ids = range(len(data))
    subset = np.random.choice(ids, size=n, replace=False)
    return data[subset]


def get_dataset(pos, all_neg, hard_negative=None):
    pos_y = np.array([1] * len(pos))
    neg = []
    if hard_negative is not None:
        neg.extend(hard_negative)

    if len(neg) < 2 * len(pos):
        neg.extend(get_rand_subset(all_neg, 2 * len(pos) - len(neg)))

    neg = np.array(neg)
    neg_y = np.array([0] * len(neg))

    print "Number of positive examples: " + str(pos.shape[0])
    print "Number of negative examples: " + str(neg.shape[0])

    X = np.vstack([pos, neg])
    y = np.hstack([pos_y, neg_y])

    return train_test_split(X, y, test_size=0.1)


def classify(paths, pos_dir):
    features = get_features(paths)
    probs = clf.predict(features)
    ids = np.where(probs)[0]
    pos_paths = [paths[idx] for idx in ids]
    if not os.path.exists(pos_dir):
        os.makedirs(pos_dir)
    for p in tqdm.tqdm(pos_paths):
        shutil.copy(p, pos_dir)


def get_features(paths):
    features = []
    for batch in tqdm.tqdm(chunks(paths, 16)):
        imgs = [image.img_to_array(image.load_img(f, target_size=(224, 224))) for f in batch]
        imgs = np.array(imgs)
        x = preprocess_input(imgs)
        preds = model.predict(x)
        features.append(preds[:,0,0,:])
    features = np.vstack(features)
    return features


if __name__ == 'main':
    model = ResNet50(weights='imagenet', include_top=False)
    negative_features = np.load('data/neg_f_1000.npy')

    raw_data_paths = glob.glob((args.target_data + '/*.jpg').replace('//', '/'))
    if len(raw_data_paths) == 0:
        print 'No data found to categorize in path: %s' % args.target_data
        sys.exit(1)

    pos_dir = os.path.join(args.target_data, 'positives')
    neg_dir = os.path.join(args.target_data, 'negatives')

    pos_paths = glob.glob((pos_dir + '/*.jpg').replace('//', '/'))
    print 'Processing positives images...'
    pos_features = get_features(pos_paths)

    hard_neg_features = None
    if os.path.exists(neg_dir):
        print 'Processing negative images...'
        hard_neg_paths = glob.glob((neg_dir + '/*.jpg').replace('//', '/'))
        hard_neg_features = get_features(hard_neg_paths)

    clf = LinearSVC()
    X_train, X_test, y_train, y_test = get_dataset(pos_features, negative_features, hard_neg_features)

    print 'Training'
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    'Training finished. Accuracy score: %f' % accuracy_score(y_test, y_pred)

    print 'Processing target images.a.'
    target_features = get_features(pos_paths)
    classify(raw_data_paths, args.result_dir)
    print 'Done!'


