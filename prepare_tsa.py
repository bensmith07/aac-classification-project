import pandas as pd
import numpy as np

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
    
    
    # join the dataframes
    
    df = pd.merge(intakes, outcomes, on='animal_id')
    
    
    # drop variables from the original outcomes table (since by definition, they're not drivers of outcome)
    
    columns = ['monthyear_outcome', 'date_of_birth', 'outcome_subtype', 'animal_type_outcome', 'sex_upon_outcome', 'age_upon_outcome', 'breed_outcome', 'color_outcome', 'name_outcome']
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

    # drop columns not used for modeling at this time
    df = df.drop(columns=['found_location', 'name', 'animal_id', 'breed_2', 'breed_3', 'color_2'])
    # filter for only the most common outcome types
    df = df[df.outcome_type.isin(['Adoption', 'Transfer', 'Return to Owner'])]
    
    return df