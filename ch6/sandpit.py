import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv(
    'https://archive.ics.uci.edu/ml/'
    'machine-learning-databases'
    '/breast-cancer-wisconsin/wdbc.data',
    header=None
)
df.head() 
df[1].value_counts()
X = df.loc[:, 2].values
y = df.loc[:, 1].values

le = LabelEncoder()

y = le.fit_transform(y)

le.classes_

le.transform(df[1].sample(19))















