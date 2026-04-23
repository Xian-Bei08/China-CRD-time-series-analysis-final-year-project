import pandas as pd


def load_crd_data(file_path):
    return pd.read_csv(file_path)


def load_pm25_data(file_path):
    return pd.read_csv(file_path)


def load_ozone_data(file_path):
    return pd.read_csv(file_path)


def load_hap_data(file_path):
    return pd.read_csv(file_path)


def load_gdp_ageing_data(file_path):
    return pd.read_csv(file_path)

def load_health_exp_data(file_path):
    return pd.read_csv(file_path)