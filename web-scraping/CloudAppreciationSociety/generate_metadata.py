import glob
import pandas as pd
from sklearn.model_selection import train_test_split

classes = [(0, 'Ac'),
            (1, 'As'),
            (2, 'Cb'),
            (3, 'Ci'),
            (4, 'Cc'),
            (5, 'Cs'),
            (6, 'Cu'),
            (7, 'Ns'),
            (8, 'Sc'),
            (9, 'St')
            ]

# classes =   [(0, 'Ac'),
#             (1, 'Cb'),
#             (2, 'Ci'),
#             (3, 'Cu'),
#             (4, 'Sc'),
#             (5, 'St')
#             ]
data_train = pd.DataFrame(columns=['path','class','n_class'])
data_test = pd.DataFrame(columns=['path','class','n_class'])

for n_class, class_ in classes:
    data_class = pd.DataFrame(columns=['path','class','n_class'])
    paths = glob.glob('/data/mandonaire/Cloudappreciationsociety/filtered_images_data_aug/{}/*'.format(class_))
    data_class['path'] = paths
    data_class['class'] = class_
    data_class['n_class'] = n_class
    train, test = train_test_split(data_class, test_size=0.2, random_state=17)
    data_train = pd.concat([data_train, train])
    data_test = pd.concat([data_test, test])

data_train.to_csv('/data/mandonaire/Cloudappreciationsociety/filtered_images_data_aug/data_train_10c.csv', index=False)
data_test.to_csv('/data/mandonaire/Cloudappreciationsociety/filtered_images_data_aug/data_test_10c.csv', index=False)