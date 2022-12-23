import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

if __name__ == "__main__":

    market_df = fetch_openml(name="bank-marketing")
    market_df = pd.DataFrame(market_df['data'])
    market_df = market_df[['V1', 'V2', 'V3', 'V4', 'V9', 'V12', 'V16']]
    market_df = market_df.rename(columns={'V1': 'age', 'V2': 'job', 'V3': 'marital', 'V4': 'education', 'V9': 'contact', 'V12': 'duration', 'V16':'outcome'})
    market_df = market_df[(market_df.outcome == 'failure')|(market_df.outcome == 'success')]

    feature_train, feature_test, label_train, label_test = train_test_split(market_df[market_df.columns.difference(['outcome'])], market_df['outcome'], 
                                                                                    test_size=0.2, shuffle=True, stratify=market_df['outcome'], random_state=607)
    ct = ColumnTransformer([('scale', StandardScaler(), ['age', 'duration']), ('onehot', OneHotEncoder(), ['job', 'marital', 'contact'])])
    feature_train = ct.fit_transform(feature_train)
    feature_test = ct.transform(feature_test)
    gbc = RandomForestClassifier(class_weight='balanced')
    gbc.fit(feature_train, label_train)

    joblib.dump(ct, './artifacts/transformer.pkl')
    joblib.dump(gbc, './artifacts/model.pkl')
    market_df.to_parquet('./artifacts/marketing_data.pqt')

    print('Finished model training.')
