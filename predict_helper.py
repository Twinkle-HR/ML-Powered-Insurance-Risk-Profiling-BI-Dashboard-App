import pandas as pd
from datetime import datetime

# region map
region_map = {
    'Delhi': 'north', 'Punjab': 'north', 'Haryana': 'north',
    'Himachal Pradesh': 'north', 'Uttarakhand': 'north',
    'Uttar Pradesh': 'north', 'Chandigarh': 'north',
    'Jammu and Kashmir': 'north', 'Ladakh': 'north',
    'Andhra Pradesh': 'south', 'Karnataka': 'south',
    'Kerala': 'south', 'Tamil Nadu': 'south', 'Telangana': 'south',
    'Puducherry': 'south', 'Lakshadweep': 'south',
    'Andaman and Nicobar Islands': 'south',
    'Bihar': 'east', 'Jharkhand': 'east', 'Odisha': 'east',
    'West Bengal': 'east', 'Sikkim': 'east', 'Assam': 'east',
    'Arunachal Pradesh': 'east', 'Manipur': 'east', 'Meghalaya': 'east',
    'Mizoram': 'east', 'Nagaland': 'east', 'Tripura': 'east',
    'Rajasthan': 'west', 'Gujarat': 'west', 'Maharashtra': 'west',
    'Goa': 'west', 'Dadra and Nagar Haveli': 'west',
    'Daman and Diu': 'west', 'Madhya Pradesh': 'west',
    'Chhattisgarh': 'west'
}

edu_map = {
    'High School': 'High School',
    'Diploma / Polytechnic': 'Undergraduate',
    'Undergraduate': 'Undergraduate',
    'Graduate': 'Graduate',
    'Postgraduate': 'Postgraduate',
    'Doctorate / Phd': 'Postgraduate',
}

def compute_age(dob_str: str) -> int:
    birth = datetime.strptime(dob_str, "%Y-%m-%d")
    today = datetime.today()
    return today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))

def get_age_group(age: int) -> str:
    if age <= 20:
        return 'Under 21'
    elif age <= 30:
        return '21-30'
    elif age <= 40:
        return '31-40'
    elif age <= 50:
        return '41-50'
    elif age <= 60:
        return '51-60'
    elif age <= 70:
        return '61-70'
    else:
        return '70+'

def categorize_income(income):
    if income < 30000:
        return 'Low'
    elif income <= 70000:
        return 'Medium'
    else:
        return 'High'

def preprocess_input(df):
    df['age_group'] = df['age'].apply(get_age_group)
    df['education_category'] = df['education_level'].map(edu_map).fillna('Other')
    df['income_group'] = df['income_level'].apply(categorize_income)
    df['region'] = df['state'].map(region_map).fillna('unknown')
    df['premium_to_coverage_ratio'] = df['premium_amount'] / df['coverage_amount']

    return df
