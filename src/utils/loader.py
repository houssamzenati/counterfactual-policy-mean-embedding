from data.advertising import Advertising
from data.pricing import Pricing

# from data.tmc2007 import tmc2007
# from data.yeast import yeast

from estimators.poem import POEM
from estimators.cpme import CPME

from policies.continuous import ContinuousPolicy
from policies.discrete import DiscretePolicy

import numpy as np
from scipy.stats import norm
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.preprocessing import add_dummy_feature


data_dico = {
    "advertising": Advertising,
    "pricing": Pricing,
    "tmc2007": None,
    "yeast": None,
}


def get_data_by_name(name, random_seed=42):
    return data_dico[name](name=name, random_seed=random_seed)


dic_estimator = {
    "poem": POEM,
    "cpme": CPME,
}


def get_estimator_by_name(name):
    return dic_estimator[name]


policy_dic = {"continuous": ContinuousPolicy, "discrete": DiscretePolicy}


def get_policy_from_type(name):
    return policy_dic[name]


def verify_settings(settings):
    if settings["data_name"] in ["pricing", "advertising"]:
        if settings["policy_type"] != "continuous":
            raise ValueError(
                "For 'pricing' or 'advertising' datasets, the policy type should always be 'continuous'."
            )

    if settings["policy_type"] == "continuous":
        if settings["pdf_type"] not in ["log_normal", "normal"]:
            raise ValueError(
                "For 'continuous' policy type, the pdf_type should always be either 'log_normal' or 'normal'."
            )
