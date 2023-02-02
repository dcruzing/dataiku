def num_to_cat_features(df, num_features:list):
    for feature in list:
        df[feature] = df[feature].astype(str)
    return df

def bin_weeks_worked(x):
    if x <13:
        return 'less than 13 weeks'
    elif x < 26:
        return 'from 13 to 25 weeks'
    elif x < 39:
        return 'from 26 to 38 weeks'
    else:
        return '39 weeks or more'

def binarize_target(x):
    if x == ' 50000+.':
        return 1
    else:
        return 0
