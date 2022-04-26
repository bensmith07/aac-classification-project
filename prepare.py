import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
random_state = 42

def aac_prep(intakes, outcomes):

    # rename columns

    for col in intakes.columns:
        intakes = intakes.rename(columns={col: f'{col.lower().replace(" ", "_")}'})

    for col in outcomes.columns:
        outcomes = outcomes.rename(columns={col: f'{col.lower().replace(" ", "_")}'})

    cols = ['name', 'datetime', 'monthyear', 'animal_type', 'breed', 'color']
    for col in cols:
        intakes = intakes.rename(columns={col: col+'_intake'})
        outcomes = outcomes.rename(columns={col: col+'_outcome'})
    
    
    # drop outcomes that don't have a corresponding intake, and vis-versa
    
    outcomes_without_intake = outcomes.loc[~ outcomes.animal_id.isin(intakes.animal_id)]
    outcomes = outcomes.loc[~ outcomes.animal_id.isin(outcomes_without_intake.animal_id)]
    
    intakes_without_outcomes = intakes.loc[~ intakes.animal_id.isin(outcomes.animal_id)]
    intakes = intakes.loc[~ intakes.animal_id.isin(intakes_without_outcomes.animal_id)]
    
    # add information about how many times the animal has been seen previously
    # and a unique identifier for each individual stay at the shelter

    intakes['datetime_intake'] = pd.to_datetime(intakes.datetime_intake)
    intakes = intakes.sort_values('datetime_intake', ignore_index=True)

    intakes['n_previous_stays'] = intakes.groupby('animal_id').cumcount()
    intakes['stay_id'] = intakes.animal_id + '_' + intakes.n_previous_stays.astype(str)

    outcomes['datetime_outcome'] = pd.to_datetime(outcomes.datetime_outcome)
    outcomes = outcomes.sort_values('datetime_outcome', ignore_index=True)

    outcomes['n_previous_stays'] = outcomes.groupby('animal_id').cumcount()
    outcomes['stay_id'] = outcomes.animal_id + '_' + outcomes.n_previous_stays.astype(str)

    # join the dataframes on stay_id
    
    df = pd.merge(intakes, outcomes, on='stay_id', suffixes=(None, '_y'))
    df = df.drop(columns=[col for col in df.columns if '_y' in col])    
    
    # drop variables from the original outcomes table (since by definition, they're not drivers of outcome)
    
    columns = ['datetime_outcome', 'monthyear_outcome', 'date_of_birth', 'outcome_subtype', 'animal_type_outcome', 'sex_upon_outcome', 'age_upon_outcome', 'breed_outcome', 'color_outcome', 'name_outcome']
    df = df.drop(columns=columns)
    
    
    # split the month_year column to extract the month (proxy for time of year), then drop month_year column
    
    df['month_intake'] = df.monthyear_intake.str.split().apply(lambda row: row[0])
    df = df.drop(columns='monthyear_intake')
    
    
    # split the sex_upon_intake column into fixed = True/False and sex = male/female
    # then drop the sex_upon_intake column
    
    df['fixed'] = df.sex_upon_intake.map({'Neutered Male': True,
                                      'Spayed Female': True,
                                      'Intact Male': False,
                                      'Intact Female': False,
                                      'Unknown': 'unknown'})
    df['sex'] = df.sex_upon_intake.map({'Neutered Male': 'male',
                                      'Spayed Female': 'female',
                                      'Intact Male': 'male',
                                      'Intact Female': 'female',
                                      'Unknown': 'unknown'})
    df = df.drop(columns='sex_upon_intake')
    
    
    # rename columns
    
    df = df.rename(columns={'animal_type_intake': 'animal_type',
                            'breed_intake': 'breed', 
                            'color_intake': 'color',
                            'name_intake': 'name'})
    
    # determine if breed is 'mixed'
        # based on whether the breed description contains the word "Mix"
        # and based on whether there is more than one breed listed in the description (separated by "/")
    # create new column breed_mixed = True/False
    # then remove the word "Mix" from the breed description
    
    def check_mixed(breed):
        if 'Mix' in breed or '/' in breed:
            return True
        else:
            return False

    df['breed_mixed'] = df.breed.apply(lambda row: check_mixed(row))
    df['breed'] = df.breed.str.replace(' Mix', '')

    # split the breed description into multiple columns when there is more than one listed
    # then drop the original breed column
    
    def breed_split_1(breed):
        if len(breed.split('/')) == 1:
            return breed
        else:
            return breed.split('/')[0]

    def breed_split_2(breed):
        if len(breed.split('/')) > 1:
            return breed.split('/')[1]
        else:
            return np.nan

    def breed_split_3(breed):
        if len(breed.split('/')) > 2:
            return breed.split('/')[2]
        else:
            return np.nan

    df['breed_1'] = df.breed.apply(breed_split_1)
    df['breed_2'] = df.breed.apply(breed_split_2)
    df['breed_3'] = df.breed.apply(breed_split_3)
    df = df.drop(columns='breed')
    
    
    # split the color descriptino into multiple columns when there is more than one listed
    # then drop the original color column
    
    def color_split_1(color):
        if len(color.split('/')) == 1:
            return color
        else:
            return color.split('/')[0]

    def color_split_2(color):
        if len(color.split('/')) > 1:
            return color.split('/')[1]
        else:
            return np.nan

    df['color_1'] = df.color.apply(color_split_1)
    df['color_2'] = df.color.apply(color_split_2)
    df = df.drop(columns='color')
    
    
    # convert age column into pandas timedelta (number of days)
    
    df['age_number'] = df.age_upon_intake.str.split().apply(lambda row: int(row[0]))
    df['age_units'] = df.age_upon_intake.str.split().apply(lambda row: row[1])
    df['age_multiplier'] = df.age_units.map({'day': 1, 
                                             'days': 1, 
                                             'week': 7, 
                                             'weeks': 7,
                                             'month': 30, 
                                             'months': 30, 
                                             'year': 365, 
                                             'years': 365})
    df['age_intake'] = df.age_number * df.age_multiplier
    df['age_intake'] = df.age_intake.apply(lambda row: pd.Timedelta(days=row))
    df = df.drop(columns=['age_number', 'age_units', 'age_multiplier', 'age_upon_intake'])
    
    # convert the date & time of the intake into a pandas datetime type
    
    df['datetime_intake'] = pd.to_datetime(df.datetime_intake)

    # add a column: found_in_austin (based on whether found location includes 'Austin (TX)')
    df['found_in_austin'] = np.where(df.found_location.str.contains('Austin (TX)', regex=False), True, False)
    # add a column: found_in_travis_cty (represents found outside city limits but inside Travis county)
    df['found_in_travis'] = np.where(df.found_location.str.contains('Travis (TX)', regex=False), True, False)
    # add a column: found_outside_jurisdiction
    df['found_outside_jurisdiction'] = np.where(df.found_location == 'Outside Jurisdiction', True, False)
    # add a column: found_other
    df['found_other'] = np.where((~df.found_location.str.contains('Austin (TX)', regex=False))
                                &(~df.found_location.str.contains('Travis (TX)', regex=False))
                                &(~df.found_location.str.contains('Outside Jurisdiction', regex=False)), True, False)
    # add a column: found_district (usually identifies city, sometimes a county, sometimes outside jurisdiction)
    df['found_district'] = df.found_location.str.split().str[-2]

    # drop columns not used for modeling at this time
    df = df.drop(columns=['datetime_intake', 'found_location', 'name', 'animal_id', 
                          'breed_2', 'breed_3', 'color_2', 'found_district'])
    # filter for only the most common outcome types (also exclude 'Return to Owner)
    df = df[df.outcome_type.isin(['Adoption', 'Transfer'])]


    
    return df

def aac_get_dogs(df):
    df = df[df.animal_type == 'Dog']

    return df

def aac_prep_for_modeling(df):

    # columns to hot code
    categorical_columns = ['fixed', 'breed_mixed', 'intake_type', 'intake_condition', 
                            'animal_type', 'month_intake', 'sex', 'breed_1_reduced', 
                            'color_1_reduced']
    # hot coding dummy variables
    for col in categorical_columns:
        dummy_df = pd.get_dummies(df[col],
                                  prefix=f'enc_{df[col].name}',
                                  drop_first=True,
                                  dummy_na=False)
        df = pd.concat([df, dummy_df], axis=1)
        # drop original column
        df = df.drop(columns=col)

    # turn age_intake timedelta into float
    df['age_intake'] = df.age_intake / pd.Timedelta(days=1)    

    return df

def train_validate_test_split(df, test_size=.2, validate_size=.3, random_state=random_state):
    '''
    This function takes in a dataframe, then splits that dataframe into three separate samples
    called train, test, and validate, for use in machine learning modeling.

    Three dataframes are returned in the following order: train, test, validate. 
    
    The function also prints the size of each sample.
    '''
    # split the dataframe into train and test
    train, test = train_test_split(df, test_size=test_size, random_state=random_state)
    # further split the train dataframe into train and validate
    train, validate = train_test_split(train, test_size=validate_size, random_state=random_state)
    # print the sample size of each resulting dataframe
    print(f'train\t n = {train.shape[0]}')
    print(f'validate n = {validate.shape[0]}')
    print(f'test\t n = {test.shape[0]}')

    return train, validate, test

def scale_aac(train, validate, test, scaler_type=MinMaxScaler()):
    # identify quantitative features to scale
    quant_features = ['age_intake']
    # establish empty dataframes for storing scaled dataset
    train_scaled = pd.DataFrame(index=train.index)
    validate_scaled = pd.DataFrame(index=validate.index)
    test_scaled = pd.DataFrame(index=test.index)
    # screate and fit the scaler
    scaler = scaler_type.fit(train[quant_features])
    # adding scaled features to scaled dataframes
    train_scaled[quant_features] = scaler.transform(train[quant_features])
    validate_scaled[quant_features] = scaler.transform(validate[quant_features])
    test_scaled[quant_features] = scaler.transform(test[quant_features])
    # add 'scaled' prefix to columns
    for feature in quant_features:
        train_scaled = train_scaled.rename(columns={feature: f'scaled_{feature}'})
        validate_scaled = validate_scaled.rename(columns={feature: f'scaled_{feature}'})
        test_scaled = test_scaled.rename(columns={feature: f'scaled_{feature}'})
    # concat scaled feature columns to original train, validate, test df's
    train = pd.concat([train, train_scaled], axis=1)
    validate = pd.concat([validate, validate_scaled], axis=1)
    test = pd.concat([test, test_scaled], axis=1)

    return train, validate, test