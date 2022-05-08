import sys
import os
from typing import List, Tuple, Any, Dict
import pandas as pd
import numpy as np
import sklearn.preprocessing as sklpre

cat_feats = [
    "MSSubClass",
    "MSZoning",
    "Street",
    "Alley",
    "LotShape",
    "LandContour",
    "Utilities",
    "LotConfig",
    "LandSlope",
    "Neighborhood",
    "Condition1",
    "Condition2",
    "BldgType",
    "HouseStyle",
    "RoofStyle",
    "RoofMatl",
    "Exterior1st",
    "Exterior2nd",
    "MasVnrType",
    "ExterQual",
    "ExterCond",
    "Foundation",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "Heating",
    "HeatingQC",
    "CentralAir",
    "Electrical",
    "KitchenQual",
    "Functional",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "PavedDrive",
    "PoolQC",
    "Fence",
    "MiscFeature",
    "SaleType",
    "SaleCondition",]

cont_feats = [
    "LotFrontage",
    "LotArea",
    "OverallQual",
    "OverallCond",
    "YearBuilt",
    "YearRemodAdd",
    "MasVnrArea",
    "BsmtFinSF1",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "1stFlrSF",
    "2ndFlrSF",
    "HalfBath",
    "LowQualFinSF",
    "GrLivArea",
    "BsmtFullBath",
    "BsmtHalfBath",
    "FullBath",
    "BedroomAbvGr",
    "KitchenAbvGr",
    "TotRmsAbvGrd",
    "Fireplaces",
    "GarageYrBlt",
    "GarageCars",
    "GarageArea",
    "WoodDeckSF",
    "OpenPorchSF",
    "EnclosedPorch",
    "3SsnPorch",
    "ScreenPorch",
    "PoolArea",
    "MiscVal",
    "MoSold",
    "YrSold",]

def encode_categorical(
        df_train: pd.DataFrame,
        ) -> Tuple[pd.DataFrame, sklpre.OrdinalEncoder]:
    cat_map = sklpre.OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1)
    df_train[cat_feats] = (
            cat_map.fit_transform(df_train[cat_feats]))
    df_train[cat_feats] = df_train[cat_feats].fillna(0).astype(int)
    return df_train, cat_map

def scale_continuous(
        df_train: pd.DataFrame,
        ) -> Tuple[pd.DataFrame, sklpre.StandardScaler]:
    std_scl = sklpre.StandardScaler()
    df_train[cont_feats] = std_scl.fit_transform(df_train[cont_feats])
    df_train[cont_feats] = df_train[cont_feats].fillna(0)
    return df_train, std_scl

def encode_onehot(
        df_train: pd.DataFrame,
        ) -> Tuple[pd.DataFrame, sklpre.OneHotEncoder]:
    one_enc = sklpre.OneHotEncoder(
            sparse=False,
            handle_unknown="ignore")
    df_aux = pd.DataFrame(
            one_enc.fit_transform(df_train[cat_feats]),
            index=df_train.index)
    df_train.drop(cat_feats, axis=1, inplace=True)
    df_train = df_train.join(df_aux)
    return df_train, one_enc

def transform_price(
        ds: pd.DataFrame,
        epsilon: float=1e-6,
        ) -> Tuple[np.array, sklpre.StandardScaler]:
    prc_scl = sklpre.StandardScaler()
    ds["SalePrice"] = np.log(ds[["SalePrice"]] + epsilon)
    ds["SalePrice"] = prc_scl.fit_transform(ds[["SalePrice"]])
    return ds, prc_scl

def transform(df: pd.DataFrame,
              cat_map: sklpre.OrdinalEncoder,
              std_scl: sklpre.StandardScaler,
              one_enc: sklpre.OneHotEncoder) -> pd.DataFrame:
    df[cat_feats] = cat_map.transform(df[cat_feats])
    df[cat_feats] = df[cat_feats].fillna(-1).astype(int)
    df[cont_feats] = std_scl.transform(df[cont_feats])
    df[cont_feats] = df[cont_feats].fillna(0)
    df_aux = pd.DataFrame(
            one_enc.transform(df[cat_feats]),
            index=df.index)
    df.drop(cat_feats, axis=1, inplace=True)
    df = df.join(df_aux)
    return df
