import pandas as pd
from datetime import datetime

def clean_category_name(name):
    if not isinstance(name, str):
        return ""
    name = name.strip().upper().replace('\n', ' ').replace('\r', ' ').replace('  ', ' ')
    if 'COOLANT ELBOW' in name:
        return 'COOLANT ELBOW & COVERS'  # unify all COOLANT ELBOW
    elif 'FEED PUMP' in name:
        return 'FEED PUMP'
    elif 'WATER PUMP' in name:
        return 'WATER PUMP'
    elif 'PCN' in name:
        return 'PCN'
    elif 'PULLEY' in name:
        return 'PULLEY'
    return name

def extract_data(filepath):
    xl = pd.ExcelFile(filepath)
    data = []
    current_date = None

    for sheet in xl.sheet_names:
        df = xl.parse(sheet, header=None).fillna("")
        for idx, row in df.iterrows():
            row_str = " ".join(map(str, row)).upper()

            if any(k in row_str for k in ["DATE", "DAY", "SHIFT"]):
                try:
                    date_cell = row.iloc[2] if len(row) > 2 else row.iloc[0]
                    current_date = pd.to_datetime(date_cell, dayfirst=True, errors='coerce')
                    if pd.isna(current_date):
                        current_date = datetime.now().date()
                except:
                    current_date = datetime.now().date()

            if "CATEGORY" in row_str and current_date:
                for i in range(idx + 1, len(df)):
                    r = df.iloc[i]
                    if len(r) < 7:
                        continue

                    cat = clean_category_name(str(r.iloc[3]))

                    if cat in ["", "TOTAL", "CATEGORY", "GRAND TOTAL"]:
                        continue

                    plan = pd.to_numeric(r.iloc[4], errors='coerce')
                    actual = pd.to_numeric(r.iloc[5], errors='coerce')
                    failure = pd.to_numeric(r.iloc[6], errors='coerce')

                    if pd.notna(plan) and pd.notna(actual):
                        data.append({
                            "date": current_date,
                            "category": cat,
                            "plan": int(plan) if pd.notna(plan) else 0,
                            "actual": int(actual) if pd.notna(actual) else 0,
                            "failure": int(failure) if pd.notna(failure) else 0
                        })

    df_out = pd.DataFrame(data)
    print(f"EXTRACTED {len(df_out)} ROWS FROM {filepath}")
    return df_out