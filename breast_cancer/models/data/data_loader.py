import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

import tensorflow as tf


def load_image(filename, label):
    file = tf.io.read_file(filename)
    img = tf.image.decode_png(file, channels=3)

    IMG_SIZE = 224
    img = tf.image.resize_with_pad(img, IMG_SIZE, IMG_SIZE)

    return img, label


def get_data(selected_fold:int=2, batch_size=28, seed:int=0):
    '''

    The data outlines a 5-fold cross-validation strategy.

    Parameters:
    ----------

    selected_fold: int
        Fold to be selected (1-5).
    
        The 2nd fold has as the highest proportion of training images.
    
    seed: int
        Radom state for reproducability.

    Returns:
    -------

    Tuple
        train_tensor, validation_tensor, test_tensor
    '''
    classes = dict(benign=0, malignant=1)
    fold_info = pd.read_csv('./data/BreaKHis_v1/Folds.csv')
    fold_info['label'] = fold_info['filename'].str.extract('(malignant|benign)')
    selected_fold_data = fold_info.query(f'fold == @selected_fold').copy()

    train = selected_fold_data.query("mag >= 200 and grp == 'train'")
    test = selected_fold_data.query("mag >= 200 and grp == 'test'")
    train.shape, test.shape

    X_train, X_valid, y_train, y_valid = train_test_split(
        train['filename'],
        train['label'].map(classes),
        random_state=seed
    )

    train_df = pd.concat([pd.Series(X_train, name='filename'), pd.Series(y_train, name='label')], axis=1)

    class_1 = train_df[train_df['label'] == 1]
    class_0 = train_df[train_df['label'] == 0]

    class_1_downsampled = resample(class_1, replace=False, n_samples=len(class_0), random_state=seed)

    train_downsampled = pd.concat([class_0, class_1_downsampled], axis=0)

    train_downsampled = train_downsampled.sample(frac=1, random_state=seed).reset_index(drop=True)

    X_train_downsampled = train_downsampled['filename']
    y_train_downsampled = train_downsampled['label']

    X_train, X_valid, y_train, y_valid = train_test_split(X_train_downsampled, y_train_downsampled, random_state=seed)

    train_tensor = tf.data.Dataset.from_tensor_slices((X_train, y_train)) \
                    .map(load_image).batch(batch_size)

    validation_tensor = tf.data.Dataset.from_tensor_slices((X_valid, y_valid)) \
                        .map(load_image).batch(batch_size)

    test = test.sample(frac=1, random_state=seed)
    test_tensor = tf.data.Dataset.from_tensor_slices((test['filename'], test['label'] \
                .map(classes))) \
                .map(load_image).batch(batch_size)


    return train_tensor, validation_tensor, test_tensor