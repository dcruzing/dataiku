import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def plot_feature_importances(model, X_train, importance_type, n_features_max = 10):
    
    # Extract categorical feature names post transformation
    new_cat_attribs = model.named_steps["FeatureEngineering"]\
    .named_transformers_["cat"].get_feature_names_out(list(X_train.select_dtypes(include='object').columns))
    
    # calculate feature importances using named importance type
    fi_df = pd.DataFrame.from_dict(data = model.named_steps["xgboostclf"].get_booster().get_score(importance_type=importance_type), orient= 'index', columns = ['importance'])
    fi_df.reset_index(inplace = True)
    fi_df.columns = ['formatted_index', 'importance']
    
    # Some features won't be used in the model and so have no feature importance and are not in the outputs. Reformat to allow joining with all feature names
    fi_df.index = fi_df['formatted_index'].str.strip('f').astype(int)
    fi_df.drop('formatted_index',axis = 1, inplace = True)
    fi_df.index.rename('index', inplace = True)
    
    # concatenate categorical columns with numerical columns to get all columns names and join the feature importance table
    importance_df = pd.DataFrame(np.concatenate([list(X_train.select_dtypes(exclude='object').columns), new_cat_attribs])).join(fi_df)
    importance_df.columns = ['feature', 'importance']
    importance_df.fillna(0, inplace = True)
    #Work out absolute feature importance
    importance_df["abs_importance"] = importance_df["importance"].apply(lambda x: abs(x))
    #Colour code for features with positive impact on predictions - this not necessary for XGBoost as min is zero
    importance_df["colors"] = importance_df["importance"].apply(lambda x: "green" if x > 0 else "red")
    importance_df = importance_df.sort_values("abs_importance", ascending=False)



    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    sns.barplot(y="feature",
                x="abs_importance",
                data=importance_df.head(n_features_max),
            palette=importance_df.head(n_features_max)["colors"], orient='h')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=20)
    ax.set_title(f"Top {n_features_max} Features, using {importance_type}", fontsize=25)
    ax.set_xlabel("Importance", fontsize=22)
    ax.set_ylabel("Feature Name", fontsize=22)

    plt.plot()
