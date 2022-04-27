import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 100)
pd.set_option('display.precision', 3)
pd.option_context('display.max_rows', 50)

def plot_normal_condition(train):
    train['adopted'] = np.where((train.outcome_type == 'Adoption'), True, False)
    train['condition_normal'] = np.where((train.intake_condition == 'Normal'), True, False)
    plt.figure(figsize=(12,8))
    sns.barplot(data=train, x='condition_normal', y='adopted')
    plt.show()

def plot_pitbulls(train):
    train['adopted'] = np.where((train.outcome_type == 'Adoption'), True, False)
    plt.figure(figsize=(12,8))
    sns.barplot(data=train, x='is_pitbull', y='adopted')
    plt.title('Adoption Rate for Pit Bulls vs Non-Pit Bulls')
    plt.show()

def plot_black_dogs(train):
    train['adopted'] = np.where((train.outcome_type == 'Adoption'), True, False)
    plt.figure(figsize=(12,8))
    sns.barplot(data=train, x='is_black', y='adopted')
    plt.title('Adoption Rate for Black Dogs vs Other Colors')
    plt.show()

def plot_breed_groups(train):
    train['adopted'] = np.where((train.outcome_type == 'Adoption'), True, False)
    plt.figure(figsize=(12,8))
    sns.barplot(data=train, x='akc_breed_group', y='adopted')
    plt.title('Adoption Rate by AKC Breed Group')
    plt.xticks(rotation=270)
    plt.show()
