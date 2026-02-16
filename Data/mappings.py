

COLUMNS_TO_DROP_M = ["Semer", "Gender", "Education", "Country", "Ethnicity", "Legalh"]

COLUMNS_TO_ENCODE_M = [
    "Age",
    "Alcohol", "Amphet", "Amyl", "Benzos", "Caff", "Cannabis",
    "Choc", "Coke", "Crack", "Ecstasy", "Heroin", "Ketamine",
    "LSD", "Meth", "Mushrooms", "Nicotine", "VSA"
]

COLUMNS_TO_DECODE_M = [
    "Age",
    "Alcohol", "Amphet", "Amyl", "Benzos", "Caff", "Cannabis",
    "Choc", "Coke", "Crack", "Ecstasy", "Heroin", "Ketamine",
    "LSD", "Meth", "Mushrooms", "Nicotine", "VSA"
]

COLUMNS_TO_DECODE_D = [
    "Age",
    "Alcohol", "Amphet", "Amyl", "Benzos", "Caff", "Cannabis",
    "Choc", "Coke", "Crack", "Ecstasy", "Heroin", "Ketamine",
    "LSD", "Meth", "Mushrooms", "Nicotine", "VSA", "Gender", "Education", "Country"]

COLUMNS_TO_DROP_D = ["Semer", "Ethnicity", "Legalh"]

AGE_MAP = {
    -0.95197: "18-24",
    -0.07854: "25-34",
     0.49788: "35-44",
     1.09449: "45-54",
     1.82213: "55-64",
     2.59171: "65+",
}

EDUCATION_MAP = {
    -2.43591: "Left School Before 16 years",
    -1.73790: "Left School at 16 years",
    -1.43719: "Left School at 17 years",
    -1.22751: "Left School at 18 years",
    -0.61113: "Some College,No Certificate Or Degree",
    -0.05921: "Professional Certificate/ Diploma",
     0.45468: "University Degree",
     1.16365: "Masters Degree",
     1.98437: "Doctorate Degree",
}

GENDER_MAP = {
     0.48246: "Female",
    -0.48246: "Male",
}

COUNTRY_MAP = {
    -0.09765: "Australia",
     0.24923: "Canada",
    -0.46841: "New Zealand",
    -0.28519: "Other",
     0.21128: "Republic of Ireland",
     0.96082: "UK",
    -0.57009: "USA",
}


DRUG_CLASS_MAP = {
    "CL0": "Never Used",
    "CL1": "Used over a Decade Ago",
    "CL2": "Used in Last Decade",
    "CL3": "Used in Last Year",
    "CL4": "Used in Last Month",
    "CL5": "Used in Last Week",
    "CL6": "Used in Last Day",
}

ALCOHOL_MAP = DRUG_CLASS_MAP
AMPHET_MAP = DRUG_CLASS_MAP
AMYL_MAP = DRUG_CLASS_MAP
BENZOS_MAP = DRUG_CLASS_MAP
CAFF_MAP = DRUG_CLASS_MAP
CANNABIS_MAP = DRUG_CLASS_MAP
CHOC_MAP = DRUG_CLASS_MAP
COKE_MAP = DRUG_CLASS_MAP
CRACK_MAP = DRUG_CLASS_MAP
ECSTASY_MAP = DRUG_CLASS_MAP
HEROIN_MAP = DRUG_CLASS_MAP
KETAMINE_MAP = DRUG_CLASS_MAP
LSD_MAP = DRUG_CLASS_MAP
METH_MAP = DRUG_CLASS_MAP
MUSHROOMS_MAP = DRUG_CLASS_MAP
NICOTINE_MAP = DRUG_CLASS_MAP
VSA_MAP = DRUG_CLASS_MAP

DECODE_MAPS_D = {
    "Age": AGE_MAP,
    "Gender": GENDER_MAP,
    "Education":EDUCATION_MAP,
    "Country": COUNTRY_MAP,

    "Alcohol": ALCOHOL_MAP,
    "Amphet": AMPHET_MAP,
    "Amyl": AMYL_MAP,
    "Benzos": BENZOS_MAP,
    "Caff": CAFF_MAP,
    "Cannabis": CANNABIS_MAP,

    "Choc": CHOC_MAP,
    "Coke": COKE_MAP,
    "Crack": CRACK_MAP,
    "Ecstasy": ECSTASY_MAP,
    "Heroin": HEROIN_MAP,
    "Ketamine": KETAMINE_MAP,
    "LSD": LSD_MAP,
    "Meth": METH_MAP,
    "Mushrooms": MUSHROOMS_MAP,
    "Nicotine": NICOTINE_MAP,
    "VSA": VSA_MAP,
}

DECODE_MAPS_M = {
    "Age": AGE_MAP,

    "Alcohol": ALCOHOL_MAP,
    "Amphet": AMPHET_MAP,
    "Amyl": AMYL_MAP,
    "Benzos": BENZOS_MAP,
    "Caff": CAFF_MAP,
    "Cannabis": CANNABIS_MAP,

    "Choc": CHOC_MAP,
    "Coke": COKE_MAP,
    "Crack": CRACK_MAP,
    "Ecstasy": ECSTASY_MAP,
    "Heroin": HEROIN_MAP,
    "Ketamine": KETAMINE_MAP,

    "LSD": LSD_MAP,
    "Meth": METH_MAP,
    "Mushrooms": MUSHROOMS_MAP,
    "Nicotine": NICOTINE_MAP,
    "VSA": VSA_MAP,
}