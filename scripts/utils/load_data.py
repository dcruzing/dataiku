import gdown
import pandas as pd

def fetch_gdrive_data(url, output):
    gdown.download(url, output, quiet=False)
    
    
def load_data_and_clean_column_names(data, metadata):
    df = pd.read_csv(data)
    column_names = pd.read_csv(metadata,
                               sep='|',
                               header = None
                               )[1].iloc[23:68].str.split('\t',
                                                          n = 1,
                                                          expand = True
                                                          )[0].str.lstrip().str.replace("'","")
    column_names = column_names[(column_names != 'adjusted gross income') &
                                (column_names != 'federal income tax liability') &
                                (column_names != 'total person earnings') & 
                                (column_names != 'total person income') &
                                (column_names != 'taxable income amount')].reset_index(drop = True)
    column_names = list(column_names)
    column_names[10] = 'race'
    column_names.extend(['year', 'income'])
    df.columns = column_names
    return df