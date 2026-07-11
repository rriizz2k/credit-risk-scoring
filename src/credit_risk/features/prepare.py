def split_features_target(df):
    df['class'] = df['class'].map({"good":0, "bad":1})

    x = df.drop(columns=['class'])
    y = df['class']


    return x, y
