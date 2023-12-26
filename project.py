# project.py


import numpy as np
import pandas as pd
import pathlib
from pathlib import Path

# If this import errors, run `pip install plotly` in your Terminal with your conda environment activated.
import plotly.express as px



# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def count_monotonic(arr):
    return (np.ediff1d(arr) < 0).sum()

def monotonic_violations_by_country(vacs): 
    # Groups by Country_Region column and aggregates by count_monotonic function. 
    monotonic_violiations = (vacs
                             .groupby('Country_Region')[['Doses_admin', 
                                                         'People_at_least_one_dose']]
                             .agg(count_monotonic))
    
    # Renames the columns into acceptable form for question.
    return (monotonic_violiations
            .rename(columns={'Doses_admin': 'Doses_admin_monotonic', 
                             'People_at_least_one_dose': 'People_at_least_one_dose_monotonic'}))


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def robust_totals(vacs):
    # Groups by Country_region and gets the 97th percentile data for respective columns.
    return (vacs
            .groupby('Country_Region')[['Doses_admin', 'People_at_least_one_dose']]
            .quantile(0.97))


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def fix_dtypes(pops_raw):
    # Convert __% to decimal
    def percent_to_decimal(percent):
        return round(float(percent[:-1]) * .01, 4)
    # Convert __/Km² to __
    def remove_Km(string):
        return float(string[:-4].replace('_', '').replace(',', ''))
    # Create a copy and data clean
    df = pops_raw.copy()
    df['World Percentage'] = df['World Percentage'].apply(percent_to_decimal)
    df['Population in 2023'] = (df['Population in 2023'] * 1000).astype('int64')
    df['Area (Km²)'] = df['Area (Km²)'].apply(remove_Km).astype('int64')
    df['Density (P/Km²)'] = df['Density (P/Km²)'].apply(remove_Km)
    return df


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def missing_in_pops(tots, pops):
    # Use the .isin() method to determine if values in pops' column is in 
    # tots' index. Then, query to get only True values after converting False to 
    # True and True to False. Convert into a set in the end.
    return set(np.array(tots
                        .index[~pd.Series(tots.index)
                               .isin(pops['Country (or dependency)'])])
                               .flatten())

    
def fix_names(pops):
    # Conversion map
    convert = {'Myanmar': 'Burma', 
               'Cape Verde': 'Cabo Verde', 
               'Republic of the Congo': 'Congo (Brazzaville)',
               'DR Congo': 'Congo (Kinshasa)', 
               "Ivory Coast": "Cote d'Ivorie", 
               'Czech Republic': 'Czechia', 
               'South Korea': 'Korea, South', 
               'United States': 'US', 
               'Palestine': 'West Bank and Gaza'}
    
    # Convert column
    df = pops.copy()
    df['Country (or dependency)'] = (df['Country (or dependency)']
                                     .replace(convert))
    return df


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def draw_choropleth(tots, pops_fixed):
    # create copy of tots DataFrame with its index reset.
    df = tots.copy().reset_index()
    # inner merge pops_fixed with tots with their chared columns.
    df_merged = pops_fixed.merge(df,
                                 how='inner', 
                                 left_on='Country (or dependency)', 
                                 right_on='Country_Region')
    df_merged['Doses Per Person'] = df_merged['Doses_admin']/df_merged['Population in 2023']
    fig = px.choropleth(df_merged, 
                        locations='ISO',  
                        hover_name='Country (or dependency)',
                        hover_data='Doses Per Person', 
                        color='Doses Per Person', 
                        range_color=(0, 4), 
                        color_continuous_scale=['white', 'green'], 
                        projection='equirectangular', 
                        title='COVID Vaccine Doses Per Person (as of 01/20/2023)', 
                        width=1000,
                        height=800)
    return fig


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def clean_israel_data(df):
    # replaces missing values in Age with 0 and changes type to float64 values.
    df['Age'] = df['Age'].replace({'-': np.NaN}).astype('float64')

    # replaces 0 and 1 in Vaccinated and Severe Sickness column to boolean values.
    df['Vaccinated'] = df['Vaccinated'].replace({0: False, 1: True})
    df['Severe Sickness'] = df['Severe Sickness'].replace({0: False, 1: True})

    return df




# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def mcar_permutation_tests(df, n_permutations=100):
    df_mcar = df.copy()
    df_mcar['Age missing'] = df_mcar['Age'].isna()

    abs_diffs_vacs = np.array([])
    abs_diffs_sicks = np.array([])
    for _ in range(n_permutations):
        df_mcar['Vaccinated'] = np.random.permutation(df_mcar['Vaccinated'])
        df_mcar['Severe Sickness'] = np.random.permutation(df_mcar['Severe Sickness'])
        
        # Computing and storing the abs_diff.
        pivoted = (
            df_mcar
            .pivot_table(index='Vaccinated', columns='Age missing', aggfunc='size')
            .apply(lambda x: x / x.sum())
        )
        
        abs_diff_vac = pivoted.diff(axis=1).iloc[:, -1].abs()[0]
        abs_diffs_vacs = np.append(abs_diffs_vacs, abs_diff_vac)

        pivoted = (
            df_mcar
            .pivot_table(index='Severe Sickness', columns='Age missing', aggfunc='size')
            .apply(lambda x: x / x.sum())
        )

        abs_diff_sick = pivoted.diff(axis=1).iloc[:, -1].abs()[0]
        abs_diffs_sicks = np.append(abs_diffs_sicks, abs_diff_sick)

    return [abs_diffs_vacs, abs_diffs_sicks]
    
    
def missingness_type():
    return 3


# ---------------------------------------------------------------------
# QUESTION 8
# ---------------------------------------------------------------------


def effectiveness(df):
    pV = df.groupby('Vaccinated')['Severe Sickness'].mean().loc[True]
    pU = df.groupby('Vaccinated')['Severe Sickness'].mean().loc[False]
    return 1 - (pV / pU)


# ---------------------------------------------------------------------
# QUESTION 9
# ---------------------------------------------------------------------


AGE_GROUPS = [
    '12-15',
    '16-19',
    '20-29',
    '30-39',
    '40-49',
    '50-59',
    '60-69',
    '70-79',
    '80-89',
    '90-'
]

def stratified_effectiveness(df):
    def series_effectiveness(series):
        return 1 - series.loc[True]/series.loc[False]

    temp_df = df.copy()
    bins = [12, 15, 19, 29, 39, 49, 59, 69, 79, 89, np.inf]
    temp_df.loc[:, ['Age Group']] = pd.cut(temp_df['Age'], bins=bins, right=True, labels=AGE_GROUPS, include_lowest=True)
    temp_df = temp_df.groupby(['Age Group', 'Vaccinated'])['Severe Sickness'].mean()

    data = np.array([series_effectiveness(temp_df.loc['12-15']), 
                     series_effectiveness(temp_df.loc['16-19']), 
                     series_effectiveness(temp_df.loc['20-29']), 
                     series_effectiveness(temp_df.loc['30-39']), 
                     series_effectiveness(temp_df.loc['40-49']), 
                     series_effectiveness(temp_df.loc['50-59']), 
                     series_effectiveness(temp_df.loc['60-69']), 
                     series_effectiveness(temp_df.loc['70-79']), 
                     series_effectiveness(temp_df.loc['80-89']), 
                     series_effectiveness(temp_df.loc['90-'])])

    return pd.Series(data, index=AGE_GROUPS)
    
    

# ---------------------------------------------------------------------
# QUESTION 10
# ---------------------------------------------------------------------


def effectiveness_calculator(
    *,
    young_vaccinated_prop,
    old_vaccinated_prop,
    young_risk_vaccinated,
    young_risk_unvaccinated,
    old_risk_vaccinated,
    old_risk_unvaccinated
):
    # Intuitive arithmetic.
    young_effectiveness = 1 - young_risk_vaccinated/young_risk_unvaccinated
    old_effectiveness = 1 - old_risk_vaccinated/old_risk_unvaccinated
    overall_prop_vaccine_illness = ((0.5 * 
                                    young_vaccinated_prop * 
                                    young_risk_vaccinated) + 
                                    (0.5 * 
                                        old_vaccinated_prop * 
                                        old_risk_vaccinated
                                    )) / (0.5 * young_vaccinated_prop + 
                                          0.5 * old_vaccinated_prop
                                        )
    overall_prop_novaccine_illness = (((0.5 * (1 - young_vaccinated_prop) * young_risk_unvaccinated) + 
                                       (0.5 * (1 - old_vaccinated_prop) * old_risk_unvaccinated)) /
                                       ((0.5 * (1 - young_vaccinated_prop) + 0.5 * (1 - old_vaccinated_prop)))) 
    overall_effectiveness = 1 - overall_prop_vaccine_illness/overall_prop_novaccine_illness
    return {'Overall': overall_effectiveness, 
            'Young': young_effectiveness, 
            'Old': old_effectiveness}


# ---------------------------------------------------------------------
# QUESTION 11
# ---------------------------------------------------------------------


def extreme_example():
    return {'young_vaccinated_prop': 0.01,
            'old_vaccinated_prop': 0.99,
            'young_risk_vaccinated': 0.03,
            'young_risk_unvaccinated': 0.16,
            'old_risk_vaccinated': 0.18,
            'old_risk_unvaccinated': 0.93}
