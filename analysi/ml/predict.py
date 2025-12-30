import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from xgboost import XGBRegressor, XGBClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import re

# --- your categories ---
ALL_CATEGORIES = ['FEED PUMP', 'WATER PUMP', 'PCN', 'COOLANT ELBOW & COVERS', 'PULLEY']

def _norm(x):
    try:
        return str(x).strip()
    except:
        return ""

# ----------------------------
# EXTRACT FINANCIAL & CAPACITY DATA FROM TABLE 1
# ----------------------------
def extract_financial_capacity_data(df_raw):
    """
    Extract financial and capacity data from the top table in Excel
    Returns dictionary with date as key and financial/capacity data as value
    """
    if df_raw is None:
        return {}
    
    df = df_raw.copy().fillna("")
    financial_capacity_data = {}
    current_date = None
    nrows, ncols = df.shape
    
    print(f"🔍 Extracting financial & capacity data from {nrows} rows")
    
    i = 0
    while i < nrows:
        # Check for date patterns like "3/3/2025", "4.3.25", "5/3/2025"
        first_cell = _norm(df.iat[i, 0])
        date_match = re.match(r'(\d+)[/\.](\d+)[/\.](\d+)', first_cell)
        
        if date_match:
            day, month, year = date_match.groups()
            if len(year) == 2:
                year = "20" + year
            
            try:
                current_date = f"{year}-{int(month):02d}-{int(day):02d}"
                print(f"📅 Processing financial data for date: {current_date}")
                
                # Look for "DESCRIPTION" header which marks the start of Table 1
                for j in range(i+1, min(i+50, nrows)):
                    desc_cell = _norm(df.iat[j, 0])
                    if "DESCRIPTION" in desc_cell.upper():
                        # Found the start of Table 1
                        start_row = j
                        
                        # Extract category headers (column names)
                        category_headers = {}
                        for col in range(1, ncols):
                            header_cell = _norm(df.iat[start_row, col])
                            if header_cell and header_cell.upper() in ALL_CATEGORIES:
                                category_headers[col] = header_cell.upper()
                        
                        # Now extract the data rows for each category
                        daily_financial_data = {}
                        
                        # Scan rows after DESCRIPTION for Operational, Capacity, Financial data
                        for row in range(start_row + 1, min(start_row + 50, nrows)):
                            row_type = _norm(df.iat[row, 0]).upper()
                            
                            # Skip empty rows
                            if not row_type:
                                continue
                            
                            # Check if we've reached the end of Table 1 (blank row or next section)
                            if row_type in ["CATEGORY", "SALARY", ""]:
                                break
                            
                            # Process Operational data
                            if row_type == "OPERATIONAL":
                                # Next rows contain operational metrics
                                for op_row in range(row + 1, row + 10):
                                    metric = _norm(df.iat[op_row, 0])
                                    if not metric or metric.upper() in ["CAPACITY", "FINANCIAL", ""]:
                                        break
                                    
                                    for col, category in category_headers.items():
                                        value = _norm(df.iat[op_row, col])
                                        if value and value != "-" and value.replace('.', '', 1).isdigit():
                                            key = f"operational_{metric.lower().replace(' ', '_')}"
                                            if category not in daily_financial_data:
                                                daily_financial_data[category] = {}
                                            daily_financial_data[category][key] = float(value)
                            
                            # Process Capacity data
                            elif row_type == "CAPACITY":
                                for cap_row in range(row + 1, row + 10):
                                    metric = _norm(df.iat[cap_row, 0])
                                    if not metric or metric.upper() in ["FINANCIAL", ""]:
                                        break
                                    
                                    for col, category in category_headers.items():
                                        value = _norm(df.iat[cap_row, col])
                                        if value and value != "-" and value.replace(',', '').replace('.', '', 1).isdigit():
                                            key = f"capacity_{metric.lower().replace(' ', '_')}"
                                            if category not in daily_financial_data:
                                                daily_financial_data[category] = {}
                                            daily_financial_data[category][key] = float(value.replace(',', ''))
                            
                            # Process Financial data
                            elif row_type == "FINANCIAL":
                                for fin_row in range(row + 1, row + 10):
                                    metric = _norm(df.iat[fin_row, 0])
                                    if not metric or metric == "":
                                        break
                                    
                                    for col, category in category_headers.items():
                                        value = _norm(df.iat[fin_row, col])
                                        if value and value != "-" and value != ".":
                                            # Clean financial values (remove commas, etc.)
                                            clean_value = value.replace(',', '').replace('₹', '').replace('$', '').strip()
                                            if clean_value and clean_value.replace('.', '', 1).isdigit():
                                                key = f"financial_{metric.lower().replace(' ', '_')}"
                                                if category not in daily_financial_data:
                                                    daily_financial_data[category] = {}
                                                daily_financial_data[category][key] = float(clean_value)
                        
                        # Store the extracted data
                        if daily_financial_data:
                            financial_capacity_data[current_date] = daily_financial_data
                            print(f"  ✅ Extracted financial data for {len(daily_financial_data)} categories")
                        
                        break  # Break out of the DESCRIPTION search loop
                
                # Skip ahead to process next date block
                i = j if 'j' in locals() else i + 1
                continue
                
            except Exception as e:
                print(f"  ⚠️ Error processing date {first_cell}: {e}")
        
        i += 1
    
    print(f"📊 Extracted financial data for {len(financial_capacity_data)} dates")
    return financial_capacity_data

# ----------------------------
# EXTRACT PRODUCTION DATA FROM TABLE 2
# ----------------------------
def extract_production_data(df_raw):
    """
    Extract production data (PLAN, ACTUAL, FAILURE) from the bottom table
    Returns list of production records
    """
    if df_raw is None:
        return []
    
    df = df_raw.copy().fillna("")
    production_data = []
    current_date = None
    nrows, ncols = df.shape
    
    print(f"🔍 Extracting production data from {nrows} rows")
    
    for i in range(nrows):
        # Check for date patterns
        first_cell = _norm(df.iat[i, 0])
        date_match = re.match(r'(\d+)[/\.](\d+)[/\.](\d+)', first_cell)
        
        if date_match:
            day, month, year = date_match.groups()
            if len(year) == 2:
                year = "20" + year
            
            try:
                current_date = f"{year}-{int(month):02d}-{int(day):02d}"
                
                # Look for "CATEGORY" header which marks the start of Table 2
                for j in range(i+1, min(i+20, nrows)):
                    cat_cell = _norm(df.iat[j, 0])
                    if "CATEGORY" in cat_cell.upper():
                        # Found the start of Table 2
                        # Now extract production data for each category
                        for k in range(j+1, min(j+10, nrows)):
                            category_cell = _norm(df.iat[k, 0])
                            
                            # Check if this is a valid category row
                            category = None
                            for cat in ALL_CATEGORIES:
                                if cat in category_cell.upper():
                                    category = cat
                                    break
                            
                            if not category:
                                # Might be a total row or blank row - skip
                                continue
                            
                            # Extract PLAN, ACTUAL, FAILURE values
                            plan_val = 0
                            actual_val = 0
                            failure_val = 0
                            
                            # Try columns 1, 2, 3 for PLAN, ACTUAL, FAILURE
                            for col_idx, value_name in [(1, "plan"), (2, "actual"), (3, "failure")]:
                                if col_idx < ncols:
                                    cell_val = _norm(df.iat[k, col_idx])
                                    if cell_val and cell_val != "-":
                                        clean_val = cell_val.replace(',', '').replace('₹', '').replace('$', '').strip()
                                        if clean_val and clean_val.replace('.', '', 1).replace('-', '', 1).isdigit():
                                            if value_name == "plan":
                                                plan_val = float(clean_val)
                                            elif value_name == "actual":
                                                actual_val = float(clean_val)
                                            elif value_name == "failure":
                                                failure_val = float(clean_val)
                            
                            # Only add if we have meaningful data
                            if plan_val > 0 or actual_val > 0 or failure_val != 0:
                                production_data.append({
                                    "date": current_date,
                                    "category": category,
                                    "plan": plan_val,
                                    "actual": actual_val,
                                    "failure": failure_val,
                                    "productive": max(actual_val, 0),
                                    "failure_units": max(abs(failure_val), 0) if failure_val < 0 else max(failure_val, 0)
                                })
                                print(f"  📊 {category}: Plan={plan_val}, Actual={actual_val}, Failure={failure_val}")
                        
                        break  # Break out of CATEGORY search loop
                
            except Exception as e:
                print(f"  ⚠️ Error processing production data for {first_cell}: {e}")
    
    print(f"📊 Extracted {len(production_data)} production records")
    return production_data

# ----------------------------
# EXTRACT REASONS (KEEP AS IS)
# ----------------------------
def extract_reasons_from_your_specific_format(df_raw):
    df = df_raw.copy().fillna('')
    category_reasons = {cat: [] for cat in ALL_CATEGORIES}
    reason_to_solution = {}
    
    df_str = df.astype(str)
    nrows, ncols = df_str.shape
    
    for r in range(nrows):
        first_cell = _norm(df_str.iat[r, 0])
        
        if re.match(r'\d+[/\.]\d+[/\.]\d+', first_cell):
            for data_row in range(r + 1, min(r + 15, nrows)):
                next_first = _norm(df_str.iat[data_row, 0])
                if re.match(r'\d+[/\.]\d+[/\.]\d+', next_first) or any(word in next_first.upper() for word in ['DESCRIPTION', 'SALARY', 'TOTAL']):
                    break
                
                category = None
                cat_cell1 = _norm(df_str.iat[data_row, 0])
                cat_cell2 = _norm(df_str.iat[data_row, 1])
                
                for cat in ALL_CATEGORIES:
                    if (cat == cat_cell1.upper().strip() or 
                        cat == cat_cell2.upper().strip() or
                        (cat in cat_cell1.upper() and len(cat_cell1) < 30) or
                        (cat in cat_cell2.upper() and len(cat_cell2) < 30)):
                        category = cat
                        break
                
                if category:
                    reason_text = ""
                    
                    if ncols > 0:
                        method_cell = _norm(df_str.iat[data_row, ncols-1])
                        if method_cell and method_cell.strip() and not _is_numeric_or_header(method_cell):
                            reason_text = method_cell
                    
                    if not reason_text:
                        header_row = r
                        for col in range(ncols):
                            header_cell = _norm(df_str.iat[header_row, col])
                            if 'REASON' in header_cell.upper():
                                reason_cell = _norm(df_str.iat[data_row, col])
                                if reason_cell and reason_cell.strip() and not _is_numeric_or_header(reason_cell):
                                    reason_text = reason_cell
                                    break
                    
                    if reason_text and reason_text.strip():
                        clean_reason = _clean_reason_text(reason_text)
                        if clean_reason and clean_reason not in category_reasons[category]:
                            category_reasons[category].append(clean_reason)
                            if clean_reason not in reason_to_solution:
                                reason_to_solution[clean_reason] = _generate_solution_from_reason(clean_reason)

    _add_fallback_reasons(category_reasons, reason_to_solution)
    
    return category_reasons, reason_to_solution, {}

def _is_numeric_or_header(text):
    if not text:
        return True
    clean_text = text.replace(',', '').replace('.', '').replace('-', '').strip()
    if clean_text.isdigit():
        return True
    if any(word in text.upper() for word in ['PLAN', 'ACTUAL', 'FAILURE', 'WAREHOUSE', 'VALUE', 'MAN', 'MACHINE', 'METHOD']):
        return True
    return False

def _clean_reason_text(reason_text):
    if not reason_text:
        return ""
    
    clean_text = reason_text.strip()
    clean_text = re.sub(r'^[\"\']|[\"\']$', '', clean_text)
    clean_text = re.sub(r'\b\d+-\s*\d+\s*NOS?\b', '', clean_text, flags=re.IGNORECASE)
    clean_text = re.sub(r'IR\s+done.*$', '', clean_text, flags=re.IGNORECASE)
    clean_text = re.sub(r'in\s+today.*$', '', clean_text, flags=re.IGNORECASE)
    clean_text = re.sub(r'\s+', ' ', clean_text)
    clean_text = clean_text.strip()
    
    if not clean_text or clean_text.replace(',', '').replace('.', '').replace('-', '').strip().isdigit():
        return ""
    
    return clean_text

def _add_fallback_reasons(category_reasons, reason_to_solution):
    fallback_reasons = {
        'FEED PUMP': [
            "IR PENDING", 
            "PACKING AND MOVING", 
            "LASER MARKING DELAY",
            "01660400- 50 Nos"
        ],
        'WATER PUMP': [
            "SEAL STOCK ISSUE", 
            "IMPELLOR SAP STOCK ISSUE", 
            "HOUSING GRN ISSUE",
            "DELAY for kitting, plan got changed & 35050000 assembly done",
            " IR delay from inhouse "
        ],
        'PCN': [
            "CLEANING PROCESS", 
            "CAVITY TRAY UNAVAILABLE", 
            "5S ACTIVITY",
            "20 Nos reworl & rejection"
        ],
        'COOLANT ELBOW & COVERS': [
            "POSITION PROBLEM", 
            "O-RING STOCK ISSUE", 
            "TOOL MARK ISSUE",
            "Fixture not yet got okay "
        ],
        'PULLEY': [
            "KANBAN NOT ISSUED", 
            "OD O/S ISSUE", 
            "INSPECTION PENDING"
        ]
    }
    
    for cat in ALL_CATEGORIES:
        if not category_reasons[cat]:
            category_reasons[cat] = fallback_reasons.get(cat, ["PRODUCTION DELAY"])
            for reason in category_reasons[cat]:
                if reason not in reason_to_solution:
                    reason_to_solution[reason] = _generate_solution_from_reason(reason)

def _generate_solution_from_reason(reason):
    reason_lower = reason.lower()
    
    solution_map = {
        'ir pending': "Expedite inspection report + Quality team follow-up",
        'seal stock': "Expedite seal stock + Supplier coordination",
        'packing': "Streamline packing process + Logistics coordination",
        'impellor': "Expedite impellor supply + SAP stock update",
        'cleaning': "Optimize cleaning schedule + Resource planning",
        'position problem': "Fixture calibration + Quality check",
        'cavity tray': "Expedite cavity tray availability",
        'kanban': "Release kanban card + Planning coordination",
        'laser marking': "Alternative marking arrangement",
        'housing grn': "Resolve GRN issue + Stores coordination",
        'od o/s': "Quality control + Segregation process",
        '5s activity': "Schedule optimization + Minimize production impact",
        'tool mark': "Tool maintenance + Process optimization",
        'o-ring stock': "Expedite O-ring supply + Buffer stock",
        'rework': "Root cause analysis + Quality review"
    }
    
    for key, solution in solution_map.items():
        if key in reason_lower:
            return solution
    
    return "Process review + Team coordination + Preventive action"

# ----------------------------
# MAIN TRAIN_MODELS FUNCTION
# ----------------------------
def train_models(df, df_raw=None):
    if df.empty:
        return {"error": "No data!"}

    # EXTRACT REASONS FROM RAW DATA
    if df_raw is not None:
        CATEGORY_REASONS, REASON_TO_SOLUTION, REASON_PROFIT_MAP = extract_reasons_from_your_specific_format(df_raw)
        # ✅ EXTRACT FINANCIAL & CAPACITY DATA FROM TABLE 1
        financial_capacity_data = extract_financial_capacity_data(df_raw)
        # ✅ EXTRACT PRODUCTION DATA FROM TABLE 2
        production_data = extract_production_data(df_raw)
    else:
        CATEGORY_REASONS = {cat: [] for cat in ALL_CATEGORIES}
        REASON_TO_SOLUTION = {}
        financial_capacity_data = {}
        production_data = []
        _add_fallback_reasons(CATEGORY_REASONS, REASON_TO_SOLUTION)

    # Common preprocessing for ML models
    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['date'])
    if df.empty:
        return {"error": "No valid dates!"}

    df = df.dropna(subset=['plan', 'actual', 'failure']).copy()
    df['plan'] = pd.to_numeric(df['plan'], errors='coerce').fillna(0).astype(int).abs()
    df['actual'] = pd.to_numeric(df['actual'], errors='coerce').fillna(0).astype(int).abs()
    df['failure'] = pd.to_numeric(df['failure'], errors='coerce').fillna(0).astype(int).abs()

    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek

    # ✅ BUILD PROFIT PER UNIT & CONVERSION COST FROM EXCEL
    category_unit_profit = {cat: 0.0 for cat in ALL_CATEGORIES}
    category_conversion_cost = {cat: 0.0 for cat in ALL_CATEGORIES}

    if df_raw is not None:
        df_r = df_raw.copy().fillna("")
        nrows, ncols = df_r.shape

        # ---------- profit per unit from VALUE / FAILURE ----------
        sums = {}  # cat -> (total_failure, total_value)

        for idx in range(nrows):
            row = df_r.iloc[idx]
            row_str = [str(c).strip().upper() for c in row]

            if "CATEGORY" in row_str:
                cat_col = fail_col = value_col = None
                for j, val in enumerate(row_str):
                    if val == "CATEGORY":
                        cat_col = j
                    elif val == "FAILURE":
                        fail_col = j
                    elif val == "VALUE":
                        value_col = j

                if cat_col is None or fail_col is None or value_col is None:
                    continue

                for r in range(idx + 1, nrows):
                    cat_raw = str(df_r.iat[r, cat_col]).strip().upper()
                    if cat_raw == "" or cat_raw in ["CATEGORY", "TOTAL", "GRAND TOTAL"]:
                        break
                    if cat_raw not in ALL_CATEGORIES:
                        continue

                    failure_raw = str(df_r.iat[r, fail_col]).replace(",", "").strip()
                    value_raw = str(df_r.iat[r, value_col]).replace(",", "").strip()

                    try:
                        failure_val = float(failure_raw)
                        value_val = float(value_raw)
                    except:
                        continue

                    if failure_val <= 0:
                        continue

                    total_fail, total_val = sums.get(cat_raw, (0.0, 0.0))
                    sums[cat_raw] = (total_fail + failure_val, total_val + value_val)

        for cat, (tot_fail, tot_val) in sums.items():
            if tot_fail > 0:
                category_unit_profit[cat] = tot_val / tot_fail

        # ---------- Conversion Cost row ----------
        df_str = df_r.astype(str)
        header_idx = None
        col_to_cat = {}

        for i in range(nrows):
            for j in range(ncols):
                val = df_str.iat[i, j].strip().upper()
                if val in ALL_CATEGORIES:
                    header_idx = i
                    break
            if header_idx is not None:
                break

        if header_idx is not None:
            for j in range(ncols):
                val = df_str.iat[header_idx, j].strip().upper()
                if val in ALL_CATEGORIES:
                    col_to_cat[j] = val

            conv_row_idx = None
            for i in range(header_idx + 1, nrows):
                for j in range(ncols):
                    if df_str.iat[i, j].strip().upper() == "CONVERSION COST":
                        conv_row_idx = i
                        break
                if conv_row_idx is not None:
                    break

            if conv_row_idx is not None:
                for col, cat in col_to_cat.items():
                    raw = str(df_r.iat[conv_row_idx, col]).replace(",", "").strip()
                    try:
                        val = float(raw)
                    except:
                        continue
                    category_conversion_cost[cat] = val

    # ADD REASON ANALYSIS FOR COUNTING
    reason_analysis = {cat: {} for cat in ALL_CATEGORIES}
    for cat in ALL_CATEGORIES:
        reasons = CATEGORY_REASONS.get(cat, [])
        for reason in reasons:
            reason_analysis[cat][reason] = {
                'count': 0,
                'total_failure': 0,
                'total_plan': 0
            }

    # RESULTS OBJECT
    results = {
        "regressors": {"models": {}, "best": {}},
        "classifiers": {"models": {}, "best": {}, "accuracy": {}},
        "predictions": [],
        "next_month_30_predictions": [],
        "top_risks": [],
        "learned_reasons": CATEGORY_REASONS,
        "reason_analysis": {},
        "daily_profit_summary": [],
        "daily_financial_capacity_summary": []  # ✅ NEW: Financial & capacity data
    }

    # ============================================================
    # ✅ CREATE DAILY FINANCIAL & CAPACITY SUMMARY (FROM TABLE 1)
    # ============================================================
    financial_capacity_rows = []
    
    for date, categories_data in financial_capacity_data.items():
        for category, metrics in categories_data.items():
            financial_capacity_rows.append({
                "date": date,
                "category": category,
                # Operational metrics
                "unit_per_person": metrics.get("operational_unit_per_person", 0),
                "on_time_shipment": metrics.get("operational_on-time_shipment", 0),
                "dock_to_dock_days": metrics.get("operational_dock-to-dock_days", 0),
                # Capacity metrics
                "productive": metrics.get("capacity_productive", 0),
                "non_productive": metrics.get("capacity_non_productive", 0),
                "available_capacity": metrics.get("capacity_available_capacity", 0),
                # Financial metrics
                "revenue": metrics.get("financial_revenue", 0),
                "material_cost": metrics.get("financial_material_cost", 0),
                "conversion_cost": metrics.get("financial_conversion_cost", 0),
                "value_stream_gross_profit": metrics.get("financial_value_stream_gross_profit", 0)
            })
    
    results["daily_financial_capacity_summary"] = financial_capacity_rows
    print(f"✅ Created {len(financial_capacity_rows)} financial/capacity records")

    # ============================================================
    # ✅ CREATE DAILY PRODUCTION SUMMARY (FROM TABLE 2)
    # ============================================================
    daily_production_rows = []
    
    if production_data:
        print(f"📊 Processing {len(production_data)} production records")
        
        # Group by date and category to combine if there are duplicates
        data_by_date_category = {}
        
        for row in production_data:
            key = (row["date"], row["category"])
            if key not in data_by_date_category:
                data_by_date_category[key] = {
                    "date": row["date"],
                    "category": row["category"],
                    "plan": 0,
                    "actual": 0,
                    "failure": 0,
                    "productive": 0,
                    "failure_units": 0,
                    "count": 0
                }
            
            data_by_date_category[key]["plan"] += row["plan"]
            data_by_date_category[key]["actual"] += row["actual"]
            data_by_date_category[key]["failure"] += row["failure"]
            data_by_date_category[key]["productive"] += row["productive"]
            data_by_date_category[key]["failure_units"] += row["failure_units"]
            data_by_date_category[key]["count"] += 1
        
        # Convert to list and calculate profit
        for key, data in data_by_date_category.items():
            category = data["category"]
            unit_profit = category_unit_profit.get(category, 0.0)
            conversion_cost = category_conversion_cost.get(category, 0.0)
            
            # Calculate profit
            profit_value = data["failure_units"] * unit_profit
            
            # Calculate profit percentage
            if conversion_cost > 0:
                profit_percentage = (profit_value / conversion_cost) * 100
            else:
                profit_percentage = 0.0
            
            daily_production_rows.append({
                "date": data["date"],
                "category": category,
                "plan": int(data["plan"]),
                "actual": int(data["actual"]),
                "failure": int(data["failure"]),
                "productive": int(data["productive"]),
                "failure_units": int(data["failure_units"]),
                "profit": round(profit_value, 2),
                "conversion_cost": round(conversion_cost, 2),
                "profit_percentage": round(profit_percentage, 2)
            })
        
        print(f"✅ Created {len(daily_production_rows)} daily production records")
    else:
        # Fallback: Use df if no production data extracted
        print("⚠️ Using fallback method for production data")
        if not df.empty:
            grouped = df.groupby(['date', 'category'], as_index=False).agg({
                'plan': 'sum',
                'actual': 'sum',
                'failure': 'sum'
            })

            for _, row in grouped.iterrows():
                cat = row['category']
                if cat not in ALL_CATEGORIES:
                    continue

                date_val = row['date']
                plan = int(row['plan'])
                actual = int(row['actual'])
                failure_units = int(max(row['failure'], 0))
                productive_units = max(actual, 0)

                unit_profit = category_unit_profit.get(cat, 0.0)
                profit_value = failure_units * unit_profit

                conversion_cost_total = category_conversion_cost.get(cat, 0.0)
                if conversion_cost_total > 0:
                    profit_percentage = round((profit_value / conversion_cost_total) * 100, 2)
                else:
                    profit_percentage = 0.0

                daily_production_rows.append({
                    "date": date_val.strftime("%Y-%m-%d"),
                    "category": cat,
                    "plan": plan,
                    "actual": actual,
                    "failure": failure_units,
                    "productive": productive_units,
                    "failure_units": failure_units,
                    "profit": round(profit_value, 2),
                    "conversion_cost": round(conversion_cost_total, 2),
                    "profit_percentage": profit_percentage
                })

    results["daily_profit_summary"] = daily_production_rows

    # =================== EXISTING ML LOGIC (KEEP AS IS) ===================
    future_models = {}

    for cat in df['category'].unique():
        data = df[df['category'] == cat].copy()
        if len(data) < 3:
            continue

        data = data.sort_values('date')

        last_plan = max(data['plan'].max(), 1)
        last_actual = data['actual'].iloc[-1]
        last_date_part = data['date'].iloc[-1]

        data['calculated_failure'] = data['plan'] - data['actual']
        data['calculated_failure'] = data['calculated_failure'].clip(lower=0)

        data['risk'] = 'LOW'
        data.loc[data['calculated_failure'] > 50000, 'risk'] = 'HIGH'
        data.loc[(data['calculated_failure'] > 10000) & (data['calculated_failure'] <= 50000), 'risk'] = 'MED'
        data['risk_encoded'] = data['risk'].map({'LOW': 0, 'MED': 1, 'HIGH': 2})

        # REGRESSION MODELS
        X_reg = data[['plan']].values
        y_reg = data['actual'].values
        X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
        scaler = MinMaxScaler()
        X_train_rs = scaler.fit_transform(X_train_r)
        X_test_rs = scaler.transform(X_test_r)

        reg_models = {
            'XGBoost': XGBRegressor(random_state=42),
            'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42),
            'Linear': LinearRegression(),
            'Decision Tree': DecisionTreeRegressor(random_state=42)
        }
        best_reg = None
        best_mae = float('inf')
        best_reg_name = "Linear"

        for name, model in reg_models.items():
            try:
                model.fit(X_train_rs, y_train_r)
                pred = model.predict(X_test_rs)
                mae = mean_absolute_error(y_test_r, pred)
                results["regressors"]["models"][f"{cat}_{name}"] = round(mae, 3)
                if mae < best_mae:
                    best_mae = mae
                    best_reg = model
                    best_reg_name = name
            except:
                pass

        results["regressors"]["best"][cat] = best_reg_name

        # CLASSIFICATION MODELS
        X_clf = data[['plan', 'actual', 'month', 'day_of_week']].values
        y_clf = data['risk_encoded'].values
        X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

        clf_models = {
            'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
            'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42),
            'Logistic': LogisticRegression(max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=42)
        }
        best_clf = None
        best_acc = 0
        best_clf_name = "Logistic"

        for name, model in clf_models.items():
            try:
                model.fit(X_train_c, y_train_c)
                pred = model.predict(X_test_c)
                acc = accuracy_score(y_test_c, pred)
                results["classifiers"]["models"][f"{cat}_{name}"] = round(acc * 100, 2)
                if acc > best_acc:
                    best_acc = acc
                    best_clf = model
                    best_clf_name = name
            except:
                pass

        results["classifiers"]["best"][cat] = best_clf_name

        future_models[cat] = {
            'reg': best_reg,
            'clf': best_clf,
            'scaler': scaler,
            'last_plan': last_plan,
            'last_actual': last_actual,
            'last_date': last_date_part,
            'historical_data': data
        }

        # PREDICTIONS WITH COUNTING
        available_reasons = CATEGORY_REASONS.get(cat, [])
        
        for i in range(1, 8):
            future_plan = int(last_plan * (1 + 0.05 * i))
            future_plan_s = scaler.transform([[future_plan]])
            pred_actual = int(best_reg.predict(future_plan_s)[0])

            if len(data) > 3:
                actual_variation = pred_actual * random.uniform(0.95, 1.05)
                pred_actual = int(actual_variation)

            failure = calculate_failure(future_plan, pred_actual, data)
            risk_pred = ['LOW', 'MED', 'HIGH'][best_clf.predict([[future_plan, pred_actual, 7, (i % 7)]])[0]]

            if failure == 0:
                reason = "NORMAL OPERATION"
                solution = "No action required"
            else:
                available_reasons = CATEGORY_REASONS.get(cat, [])
                if available_reasons:
                    reason_index = (i - 1) % len(available_reasons)
                    reason = available_reasons[reason_index]
                    solution = REASON_TO_SOLUTION.get(reason, "Process optimization required")
                    
                    if reason in reason_analysis[cat]:
                        reason_analysis[cat][reason]['count'] += 1
                        reason_analysis[cat][reason]['total_failure'] += failure
                        reason_analysis[cat][reason]['total_plan'] += future_plan
                else:
                    reason = "PRODUCTION DELAY"
                    solution = "Process review + Team coordination"

            results["predictions"].append({
                "part": cat,
                "day": f"Day {i}",
                "plan": future_plan,
                "predicted_actual": pred_actual,
                "failure": failure,
                "risk": risk_pred,
                "reason": reason,
                "solution": solution
            })

        total_fail = data['calculated_failure'].sum()
        if total_fail > 0:
            results["top_risks"].append({
                "part": cat,
                "total": int(total_fail),
                "mae": round(best_mae, 3),
                "clf_acc": round(best_acc * 100, 2),
                "trend": round(calculate_trend(data['calculated_failure'].tolist()), 3)
            })

    results["top_risks"] = sorted(results["top_risks"], key=lambda x: x["total"], reverse=True)[:5]

    # NEXT 30 DAYS PREDICTIONS
    for cat, m in future_models.items():
        for i in range(1, 31):
            future_plan = int(m['last_plan'] * (1 + 0.05 * i))
            future_plan_s = m['scaler'].transform([[future_plan]])
            pred_actual = int(m['reg'].predict(future_plan_s)[0])

            if len(m['historical_data']) > 3:
                actual_variation = pred_actual * random.uniform(0.95, 1.05)
                pred_actual = int(actual_variation)

            failure = calculate_failure(future_plan, pred_actual, m['historical_data'])
            risk_pred = ['LOW', 'MED', 'HIGH'][m['clf'].predict([[future_plan, pred_actual, 7, (i % 7)]])[0]]

            if failure == 0:
                reason = "NORMAL OPERATION"
                solution = "No action required"
            else:
                available_reasons = CATEGORY_REASONS.get(cat, [])
                if available_reasons:
                    reason_index = (i - 1) % len(available_reasons)
                    reason = available_reasons[reason_index]
                    solution = REASON_TO_SOLUTION.get(reason, "Process optimization required")
                    
                    if reason in reason_analysis[cat]:
                        reason_analysis[cat][reason]['count'] += 1
                        reason_analysis[cat][reason]['total_failure'] += failure
                        reason_analysis[cat][reason]['total_plan'] += future_plan
                else:
                    reason = "PRODUCTION DELAY"
                    solution = "Process review + Team coordination"

            results["next_month_30_predictions"].append({
                "part": cat,
                "day": f"Day {i}",
                "plan": future_plan,
                "predicted_actual": pred_actual,
                "failure": failure,
                "risk": risk_pred,
                "reason": reason,
                "solution": solution
            })

    # CALCULATE FINAL REASON ANALYSIS
    results["reason_analysis"] = {}
    
    for cat in ALL_CATEGORIES:
        results["reason_analysis"][cat] = []
        unit_profit = category_unit_profit.get(cat, 0.0)
        
        for reason, analysis in reason_analysis[cat].items():
            if analysis['count'] > 0:
                count = analysis['count']
                predicted_production_if_solved = analysis['total_plan'] - analysis['total_failure']
                projected_increase = analysis['total_failure']
                profit_impact = projected_increase * unit_profit

                results["reason_analysis"][cat].append({
                    'reason': reason,
                    'count': count,
                    'predicted_production_if_solved': int(predicted_production_if_solved),
                    'projected_increase': int(projected_increase),
                    'profit_impact': round(profit_impact, 2)
                })
        
        results["reason_analysis"][cat] = sorted(
            results["reason_analysis"][cat], 
            key=lambda x: x['profit_impact'], 
            reverse=True
        )

    return results

def calculate_trend(values):
    if len(values) < 2:
        return 0
    try:
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return slope / np.mean(values) if np.mean(values) > 0 else 0
    except:
        return 0

def calculate_failure(future_plan, pred_actual, historical_data):
    failure = max(0, future_plan - pred_actual)
    failure = min(failure, future_plan)
    return failure