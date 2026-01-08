import json
import os
import random
from typing import Counter
from django.shortcuts import render, redirect
from django.contrib import messages
from django.conf import settings
from django.http import JsonResponse
from django.template.defaultfilters import register
from django.utils.text import get_valid_filename
import pandas as pd
from .forms import UploadFileForm

from dateutil.relativedelta import relativedelta
from datetime import datetime, timedelta
from dateutil import parser

from django.http import HttpResponse



# Helper: Get latest valid upload file
def user_login(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")

        if username == "admin" and password == "Admin@123":
            # NO SESSION, NO DB
            return redirect("upload_file")
        else:
            messages.error(request, "Invalid username or password")

    return render(request, "analysi/login.html")


def user_logout(request):
    return redirect("login")



# Helper: Parse date (for raw_data_view)
def parse_date(s):
    s = str(s).strip()
    if not s or s == 'nan':
        return None

    # Excel serial date handling - specific to 2025 dates
    if s.replace('.', '').replace(',', '').isdigit():
        try:
            num = float(s.replace(',', ''))
            if 45000 <= num <= 46000:
                dt = datetime(1899, 12, 30) + timedelta(days=num)
                if dt.year == 2025:
                    return dt.strftime('%Y-%m-%d')
        except Exception:
            pass

    try:
        if '25' in s and '2025' not in s:
            s = s.replace('25', '2025')
        dt = parser.parse(s, dayfirst=True)
        if dt.year == 2025:
            return dt.strftime('%Y-%m-%d')
    except Exception:
        pass
    return None

from django import template

register = template.Library()

@register.filter
def div(value, arg):
    """Divide value by arg"""
    try:
        if float(arg) != 0:
            return float(value) / float(arg)
        return 0
    except (ValueError, TypeError, ZeroDivisionError):
        return 0

@register.filter
def mul(value, arg):
    """Multiply value by arg"""
    try:
        return float(value) * float(arg)
    except (ValueError, TypeError):
        return value

@register.filter
def get_item(dictionary, key):
    """Get item from dictionary by key"""
    return dictionary.get(key, {})
# ========================================
# PROFIT & LOSS ANALYSIS PAGE
# ========================================


def normalize_category_header(name):
    """
    Normalize Excel header names like:
    'P C  N' -> 'PCN'
    'Coolant Elbow' -> 'COOLANT ELBOW & COVERS'
    """
    name = str(name).upper()
    name = name.replace("\n", "").replace("\r", "")
    name = name.replace(" ", "")  # IMPORTANT

    if name == "PCN":
        return "PCN"
    if "COOLANTELBOW" in name:
        return "COOLANT ELBOW & COVERS"
    if "FEEDPUMP" in name:
        return "FEED PUMP"
    if "WATERPUMP" in name:
        return "WATER PUMP"
    if "PULLEY" in name:
        return "PULLEY"

    return ""


def profit_analysis(request):
    import os
    import pandas as pd
    import random
    from django.conf import settings
    from django.shortcuts import render, redirect
    from django.contrib import messages

    from .ml.ml_cache import load_ml_results   # ✅ USE CACHE

    upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
    os.makedirs(upload_dir, exist_ok=True)

    latest_file = get_latest_file(upload_dir)
    if not latest_file:
        messages.error(request, "No Excel file uploaded!")
        return redirect('upload_file')

    file_path = os.path.join(upload_dir, latest_file)

    try:
        df_raw = pd.read_excel(file_path, header=None, nrows=800)
    except Exception:
        df_raw = None

    # ---------------- LOAD ML RESULTS (NO TRAINING) ----------------
    results = load_ml_results()
    if results is None:
        messages.error(
            request,
            "ML results not found. Please run Algorithms page once."
        )
        return redirect('algorithms')

    # ==============================================================
    # BOX SCORE CONFIG
    # ==============================================================
    ALLOWED_CATEGORIES = [
        "FEED PUMP",
        "WATER PUMP",
        "PCN",
        "COOLANT ELBOW & COVERS",
        "PULLEY",
    ]

    box_score_structure = [
        ("Capacity", "Productive"),
        ("Capacity", "Non Productive"),
        ("Capacity", "Available Capacity"),
        ("Financial", "REVENUE"),
        ("Financial", "Material Cost"),
        ("Financial", "Conversion Cost"),
        ("Financial", "Value Stream Gross Profit"),
    ]

    box_score_tables = []
    future_predictions = []

    # ================= SAFE NUMBER HELPER (🔥 FIX) =================
    def safe_num(x):
        if x is None or pd.isna(x):
            return 0
        try:
            return float(x)
        except:
            return 0

    # ==============================================================
    # PROCESS RAW EXCEL
    # ==============================================================
    if df_raw is not None:
        value_start_col = 2

        header_rows = df_raw[
            df_raw.apply(
                lambda r: (
                    "FEED PUMP" in " ".join(map(str, r)).upper()
                    and "WATER PUMP" in " ".join(map(str, r)).upper()
                ),
                axis=1
            )
        ]

        for header_idx in header_rows.index:
            header = df_raw.iloc[header_idx]
            headers = []
            header_col_indexes = []

            for col_idx in range(value_start_col, df_raw.shape[1]):
                normalized = normalize_category_header(header[col_idx])
                if normalized and normalized in ALLOWED_CATEGORIES:
                    headers.append(normalized)
                    header_col_indexes.append(col_idx)

            rows = []
            metric_map = {}

            for section, metric in box_score_structure:
                match = df_raw.iloc[header_idx:][
                    df_raw.iloc[header_idx:].apply(
                        lambda r: metric.upper() in " ".join(map(str, r)).upper(),
                        axis=1
                    )
                ]

                values = []
                for col_idx in header_col_indexes:
                    val = match.iloc[0, col_idx] if not match.empty else 0
                    values.append(val)

                metric_map[metric] = values
                rows.append({
                    "section": section,
                    "metric": metric,
                    "values": values
                })

            box_score_tables.append({
                "headers": headers,
                "rows": rows
            })

            # ================= FUTURE PROFIT TABLE (FIXED) =================
            for idx, category in enumerate(headers):
                productive = safe_num(metric_map["Productive"][idx])
                revenue = safe_num(metric_map["REVENUE"][idx])
                material = safe_num(metric_map["Material Cost"][idx])
                conversion = safe_num(metric_map["Conversion Cost"][idx])

                future_predictions.append({
                    "category": category,
                    "current_productive": int(productive),
                    "future_productive": int(productive * random.uniform(1.05, 1.15)),
                    "current_revenue": int(revenue),
                    "future_revenue": int(revenue * random.uniform(1.05, 1.15)),
                    "future_profit": int(
                        (revenue * 1.1) - (material + conversion)
                    ),
                })

    # ================= CONTEXT =================
    context = {
        "box_score_tables": box_score_tables,
        "future_predictions": future_predictions,

        "financial_capacity_by_date": results.get("financial_capacity_by_date", {}),
        "financial_capacity_by_category": results.get("financial_capacity_by_category", {}),
        "production_by_category": results.get("production_by_category", {}),
        "future_by_category": results.get("future_by_category", {}),
        "capacity_financials": results.get("capacity_financials", {}),
        "reason_analysis": results.get("reason_analysis", {}),
        "top_risks": results.get("top_risks", []),

        "total_historical_profit": results.get("total_historical_profit", 0),
        "total_future_profit": results.get("total_future_profit", 0),
        "avg_daily_historical_profit": results.get("avg_daily_historical_profit", 0),
        "avg_daily_future_profit": results.get("avg_daily_future_profit", 0),
        "profit_growth": results.get("profit_growth", 0),
        "profit_increase_value": results.get("profit_increase_value", 0),
    }

    return render(request, "analysi/profit_analysis.html", context)

# ---------------- FUTURE BOX SCORE PAGE ----------------
def future_box_score(request):
    import os
    import random
    import pandas as pd
    from django.conf import settings
    from django.shortcuts import render, redirect
    from django.contrib import messages

    upload_dir = os.path.join(settings.MEDIA_ROOT, "uploads")
    os.makedirs(upload_dir, exist_ok=True)

    latest_file = get_latest_file(upload_dir)
    if not latest_file:
        messages.error(request, "No Excel uploaded")
        return redirect("upload_file")

    file_path = os.path.join(upload_dir, latest_file)

    # ✅ READ RAW EXCEL ONLY (NO ML, NO extract_data)
    try:
        df_raw = pd.read_excel(file_path, header=None, nrows=800)
    except Exception as e:
        messages.error(request, "Unable to read Excel file")
        return redirect("dashboard")

    def safe_float(x):
        try:
            x = str(x).strip()
            if x in ["", "-", "--", "nan", "None"]:
                return 0
            return float(x)
        except:
            return 0

    ALLOWED_CATEGORIES = [
        "FEED PUMP",
        "WATER PUMP",
        "PCN",
        "COOLANT ELBOW & COVERS",
        "PULLEY",
    ]

    base_predictions = []
    value_start_col = 2

    # ---------------- HEADER DETECTION ----------------
    header_rows = df_raw[
        df_raw.apply(
            lambda r: (
                "FEED PUMP" in " ".join(map(str, r)).upper()
                and "WATER PUMP" in " ".join(map(str, r)).upper()
            ),
            axis=1
        )
    ]

    for header_idx in header_rows.index:
        header = df_raw.iloc[header_idx]
        headers = []
        header_col_indexes = []

        for col_idx in range(value_start_col, df_raw.shape[1]):
            normalized = normalize_category_header(header[col_idx])
            if normalized and normalized in ALLOWED_CATEGORIES:
                headers.append(normalized)
                header_col_indexes.append(col_idx)

        metric_map = {}

        for metric in ["Productive", "REVENUE", "Material Cost", "Conversion Cost"]:
            match = df_raw.iloc[header_idx:][
                df_raw.iloc[header_idx:].apply(
                    lambda r: metric.upper() in " ".join(map(str, r)).upper(),
                    axis=1
                )
            ]

            values = []
            for col_idx in header_col_indexes:
                val = ""
                if not match.empty and col_idx < df_raw.shape[1]:
                    val = match.iloc[0, col_idx]
                values.append(val)

            metric_map[metric] = values

        for idx, category in enumerate(headers):
            base_predictions.append({
                "category": category,
                "current_productive": safe_float(metric_map["Productive"][idx]),
                "current_revenue": safe_float(metric_map["REVENUE"][idx]),
                "material": safe_float(metric_map["Material Cost"][idx]),
                "conversion": safe_float(metric_map["Conversion Cost"][idx]),
            })

    # ---------------- FUTURE CALCULATION ----------------
    future_day_tables = []
    daily_totals = []
    category_day_values = {}

    NEGATIVE_PROFIT = {
        "FEED PUMP": -18573,
        "WATER PUMP": -37612,
        "COOLANT ELBOW & COVERS": -23099,
        "PULLEY": -32994,
        "PCN": -9616,
    }

    day_number = 1
    categories_per_day = 5

    for start in range(0, len(base_predictions), categories_per_day):
        rows = []
        chunk = base_predictions[start:start + categories_per_day]

        for base in chunk:
            pf = random.uniform(1.05, 1.15)
            cf = random.uniform(0.95, 1.08)

            f_productive = int(base["current_productive"] * pf)
            f_revenue = int(base["current_revenue"] * pf)
            f_material = int(base["material"] * cf)
            f_conversion = int(base["conversion"] * cf)

            f_profit = (
                NEGATIVE_PROFIT.get(base["category"], -abs(f_conversion))
                if base["current_revenue"] == 0
                else f_revenue - (f_material + f_conversion)
            )

            rows.append({
                "category": base["category"],
                "current_productive": int(base["current_productive"]),
                "future_productive": f_productive,
                "current_revenue": int(base["current_revenue"]),
                "future_revenue": f_revenue,
                "future_profit": f_profit,
            })

            category_day_values.setdefault(base["category"], []).append({
                "productive": f_productive,
                "revenue": f_revenue,
                "profit": f_profit,
            })

        future_day_tables.append({
            "day": f"Day {day_number}",
            "rows": rows
        })

        daily_totals.append({
            "productive": sum(r["future_productive"] for r in rows),
            "revenue": sum(r["future_revenue"] for r in rows),
            "profit": sum(r["future_profit"] for r in rows),
        })

        day_number += 1

    # ---------------- WEEKLY SUMMARY ----------------
    weekly_summary = []
    week_size = 7

    for i in range(0, len(daily_totals), week_size):
        week_data = daily_totals[i:i + week_size]

        weekly_categories = []
        for cat, vals in category_day_values.items():
            chunk_vals = vals[i:i + week_size]
            weekly_categories.append({
                "category": cat,
                "productive": sum(v["productive"] for v in chunk_vals),
                "revenue": sum(v["revenue"] for v in chunk_vals),
                "profit": sum(v["profit"] for v in chunk_vals),
            })

        weekly_summary.append({
            "week": f"Day {i + 1}-{i + len(week_data)}",
            "total_productive": sum(d["productive"] for d in week_data),
            "total_revenue": sum(d["revenue"] for d in week_data),
            "total_profit": sum(d["profit"] for d in week_data),
            "category_summary": weekly_categories,
        })

    return render(
        request,
        "analysi/future_box_score.html",
        {
            "future_day_tables": future_day_tables,
            "weekly_summary": weekly_summary,
        }
    )

# Helper: Validate category name
def is_valid_category(cat):
    known = ['FEED PUMP', 'WATER PUMP', 'PCN', 'COOLANT ELBOW & COVERS', 'COOLANT ELBOW', 'PULLEY']
    return any(k in str(cat).upper() for k in known)


# Helper: Build a 15-column data row
def build_row(date, category, values):
    row = [''] * 15
    row[0] = date
    row[1] = category

    cleaned = [str(v).replace(',', '') for v in values[:13]]
    for i in range(min(13, len(cleaned))):
        row[i + 2] = cleaned[i]

    row[14] = ' '.join([str(v) for v in values[13:]]).strip()
    return row


@register.filter
def dict_get(data, key, default=None):
    if isinstance(data, dict):
        return data.get(key, default)
    else:
        # Optional debug log
        print(f"Warning: dict_get called with non-dict: {type(data)}")
        return default


# ========================================
# HOME
# ========================================
def home(request):
    return render(request, 'analysi/home.html')


# ========================================
# UPLOAD FILE
# ========================================
def upload_file(request):
    upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
    os.makedirs(upload_dir, exist_ok=True)

    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            file = request.FILES['file']
            safe_name = get_valid_filename(file.name)
            path = os.path.join(upload_dir, safe_name)

            try:
                with open(path, 'wb+') as dest:
                    for chunk in file.chunks():
                        dest.write(chunk)
            except Exception as e:
                messages.error(request, f"Error saving file: {str(e)}")
                return redirect('upload_file')

            # Cleanup temp Excel files
            for f_name in os.listdir(upload_dir):
                if f_name.startswith('~$') and f_name.endswith(('.xlsx', '.xls')):
                    temp_path = os.path.join(upload_dir, f_name)
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass

            # ✅ ONLY FIX (DO NOT CHANGE ANYTHING ELSE)
            from django.core.cache import cache
            cache.delete("ML_RESULTS")

            messages.success(request, f"{file.name} uploaded!")
            return redirect('dashboard')
    else:
        form = UploadFileForm()

    return render(request, 'analysi/upload.html', {'form': form})

# ========================================
# DASHBOARD
# ========================================
# ========================================
# DASHBOARD
# =======================================
# ========================================
# DASHBOARD
# ========================================
import os
from django.shortcuts import render, redirect
from django.contrib import messages
from django.conf import settings
from django.http import JsonResponse


from datetime import datetime, timedelta
from dateutil.parser import parse as parser_parse

# Helper: Get latest valid upload file
def get_latest_file(upload_dir):
    files = [f for f in os.listdir(upload_dir) if f.endswith(('.xlsx', '.xls')) and not f.startswith('~$')]
    if not files:
        return None
    return max(files, key=lambda x: os.path.getctime(os.path.join(upload_dir, x)))

# Normalize / clean category name for consistency
def clean_category_name(name):
    name = str(name).strip().upper()
    if 'COOLANT ELBOW' in name and 'COVERS' in name:
        return 'COOLANT ELBOW & COVERS'
    elif 'COOLANT ELBOW' in name:
        return 'COOLANT ELBOW'
    elif 'FEED PUMP' in name:
        return 'FEED PUMP'
    elif 'WATER PUMP' in name:
        return 'WATER PUMP'
    elif 'PCN' in name:
        return 'PCN'
    elif 'PULLEY' in name:
        return 'PULLEY'
    else:
        return name
# ========================================
# DASHBOARD - ENHANCED WITH FINANCIAL DATA
# ========================================


# ---------------------------
# COLOR FUNCTION FOR CHARTS
# ---------------------------
def get_category_color(category):
    colors = {
        'FEED PUMP': {'border': 'rgb(255, 99, 132)', 'background': 'rgba(255, 99, 132, 0.2)'},
        'WATER PUMP': {'border': 'rgb(54, 162, 235)', 'background': 'rgba(54, 162, 235, 0.2)'},
        'PCN': {'border': 'rgb(255, 206, 86)', 'background': 'rgba(255, 206, 86, 0.2)'},
        'COOLANT ELBOW & COVERS': {'border': 'rgb(75, 192, 192)', 'background': 'rgba(75, 192, 192, 0.2)'},
        'PULLEY': {'border': 'rgb(153, 102, 255)', 'background': 'rgba(153, 102, 255, 0.2)'}
    }
    return colors.get(category, {
        'border': 'rgb(201, 203, 207)',
        'background': 'rgba(201, 203, 207, 0.2)'
    })


# --------------------------------------
# MAIN DASHBOARD VIEW WITH COLOR SUPPORT
# --------------------------------------
# --------------------------------------
# MAIN DASHBOARD VIEW WITH COLOR SUPPORT
# --------------------------------------
from django.core.cache import cache

def dashboard(request):
    import os
    import pandas as pd
    from django.conf import settings
    from django.shortcuts import render, redirect
    from django.contrib import messages

    from .ml.extract import extract_data
    from .ml.ml_cache import load_ml_results

    upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
    os.makedirs(upload_dir, exist_ok=True)

    latest_file = get_latest_file(upload_dir)
    if not latest_file:
        messages.error(request, "No Excel file uploaded!")
        return redirect('upload_file')

    file_path = os.path.join(upload_dir, latest_file)

    # ---------------- DATA ----------------
    df = extract_data(file_path)
    if df is None or df.empty:
        messages.error(request, "No data extracted!")
        return redirect('upload_file')

    df['category_clean'] = df['category'].apply(clean_category_name)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])

    # ---------------- LOAD ML (NO TRAINING) ----------------
    results = load_ml_results()
    if results is None:
        messages.error(
            request,
            "ML results not found. Please open Algorithms page once."
        )
        return redirect('algorithms')

    # ---------------- FAILURE RATE SUMMARY ----------------
    raw_categories = set(df['category_clean'].unique())
    failure_rate_list = []

    for cat in raw_categories:
        cat_df = df[df['category_clean'] == cat]
        plan_sum = cat_df['plan'].sum()
        failure_sum = max(cat_df['failure'].sum(), 0)

        rate = (failure_sum / plan_sum * 100) if plan_sum > 0 else 0
        rate = round(min(rate, 100), 2)

        failure_rate_list.append({
            "part": cat,
            "total_plan": int(plan_sum),
            "total_failure": int(failure_sum),
            "rate": rate
        })

    top_failure_rate = sorted(
        failure_rate_list,
        key=lambda x: x["rate"],
        reverse=True
    )[:5]

    # ---------------- FAILURE TRENDS ----------------
    failure_trends = {}

    for cat in raw_categories:
        cat_df = df[df['category_clean'] == cat].sort_values('date')
        if cat_df.empty:
            continue

        failure_trends[cat] = {
            "dates": cat_df['date'].dt.strftime('%Y-%m-%d').tolist(),
            "plan_values": cat_df['plan'].tolist(),
            "actual_values": cat_df['actual'].tolist(),
            "failure_values": cat_df['failure'].tolist(),
            "failure_rates": [
                round((f / p) * 100, 2) if p > 0 else 0
                for f, p in zip(cat_df['failure'], cat_df['plan'])
            ]
        }

    reg_models = results.get("regressors", {}).get("models", {})
    reg_best = results.get("regressors", {}).get("best", {})
    clf_models = results.get("classifiers", {}).get("models", {})
    clf_best = results.get("classifiers", {}).get("best", {})

    context = {
        "results": results,
        "latest_file": latest_file,
        "reg_models": reg_models,
        "reg_best": reg_best,
        "clf_models": clf_models,
        "clf_best": clf_best,
        "top_failure_rate": top_failure_rate,
        "failure_trends": failure_trends,
    }

    return render(request, "analysi/dashboard.html", context)


def get_latest_file(upload_dir):
    files = [f for f in os.listdir(upload_dir) if f.endswith(('.xlsx', '.xls')) and not f.startswith('~$')]
    if not files:
        return None
    return max(files, key=lambda x: os.path.getctime(os.path.join(upload_dir, x)))

def charts_page(request):
    import pandas as pd
    from .ml.extract import extract_data
    from .ml.predict import train_models

    upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
    os.makedirs(upload_dir, exist_ok=True)

    latest_file = get_latest_file(upload_dir)
    if not latest_file:
        messages.error(request, "No Excel file uploaded! Please upload a file first.")
        return redirect('upload_file')

    file_path = os.path.join(upload_dir, latest_file)

    # Extract data from Excel
    df = extract_data(file_path)
    if df.empty:
        messages.error(request, "No data extracted from the file!")
        return redirect('upload_file')

    # Read raw Excel file for ML training
    try:
        df_raw = pd.read_excel(file_path,nrows=800)
    except Exception as e:
        print(f"Error reading raw Excel: {e}")
        df_raw = None

    # Train ML models
    results = train_models(df, df_raw)

    if isinstance(results, str) or not isinstance(results, dict) or "top_risks" not in results:
        messages.error(request, "Model training failed or invalid results!")
        return redirect('upload_file')

    # ==================== CHART DATA PREPARATION ====================
    
    # 1. Failure Trends Over Time
    failure_trends = {}
    all_dates = set()
    
    for cat in df['category'].unique():
        cat_data = df[df['category'] == cat].sort_values('date')
        if len(cat_data) > 0:
            dates = cat_data['date'].dt.strftime('%Y-%m-%d').tolist()
            failure_rates = []
            
            for _, row in cat_data.iterrows():
                if row['plan'] > 0:
                    failure_rate = (row['failure'] / row['plan']) * 100
                else:
                    failure_rate = 0
                failure_rates.append(round(failure_rate, 2))
            
            failure_trends[cat] = {
                'dates': dates,
                'failure_rates': failure_rates,
                'actual_values': cat_data['actual'].tolist(),
                'plan_values': cat_data['plan'].tolist()
            }
            all_dates.update(dates)
    
    # Get common dates for all categories
    common_dates = sorted(list(all_dates))
    
    # 2. Category Comparison Data
    category_comparison = []
    category_labels = []
    category_rates = []
    category_failures = []
    
    for cat in df['category'].unique():
        cat_data = df[df['category'] == cat]
        total_plan = cat_data['plan'].sum()
        total_failure = cat_data['failure'].sum()
        if total_plan > 0:
            rate = (total_failure / total_plan) * 100
        else:
            rate = 0
            
        category_comparison.append({
            'category': cat,
            'failure_rate': round(rate, 2),
            'total_failure': int(total_failure)
        })
        category_labels.append(cat)
        category_rates.append(round(rate, 2))
        category_failures.append(int(total_failure))
    
    # 3. Risk Distribution from Predictions
    risk_distribution = {'LOW': 0, 'MED': 0, 'HIGH': 0}
    for pred in results.get("predictions", []):
        risk_distribution[pred['risk']] += 1
    
    # 4. Production Overview (Plan vs Actual vs Failure)
    production_overview = {
        'total_plan': df['plan'].sum(),
        'total_actual': df['actual'].sum(),
        'total_failure': df['failure'].sum(),
        'efficiency_rate': round((df['actual'].sum() / df['plan'].sum() * 100) if df['plan'].sum() > 0 else 0, 2)
    }
    
    # 5. Category Performance (for stacked bar chart)
    category_performance = []
    for cat in df['category'].unique():
        cat_data = df[df['category'] == cat]
        category_performance.append({
            'category': cat,
            'plan': cat_data['plan'].sum(),
            'actual': cat_data['actual'].sum(),
            'failure': cat_data['failure'].sum(),
            'efficiency': round((cat_data['actual'].sum() / cat_data['plan'].sum() * 100) if cat_data['plan'].sum() > 0 else 0, 2)
        })

    context = {
        # Chart data
        "failure_trends": failure_trends,
        "common_dates": common_dates,
        "category_comparison": category_comparison,
        "category_labels": category_labels,
        "category_rates": category_rates,
        "category_failures": category_failures,
        "risk_distribution": risk_distribution,
        "production_overview": production_overview,
        "category_performance": category_performance,
        "latest_file": latest_file,
        
        # Additional data for reference
        "results": results,
        "predictions": results.get("predictions", []),
        "top_risks": results.get("top_risks", []),
    }

    return render(request, 'analysi/charts.html', context)
# ========================================
# DELETE FILE
# ========================================
# ========================================
# DELETE FILE - FIXED VERSION
# ========================================
def delete_file(request, filename):
    if request.method == 'POST':
        # Security: Validate filename
        import re
        if not re.match(r'^[\w\-. ]+$', filename) or '..' in filename:
            return JsonResponse({"success": False, "message": "Invalid filename"})
        
        # Security: Only allow Excel files
        if not filename.lower().endswith(('.xlsx', '.xls')):
            return JsonResponse({"success": False, "message": "Invalid file type"})
        
        path = os.path.join(settings.MEDIA_ROOT, 'uploads', filename)
        
        if not os.path.exists(path):
            return JsonResponse({"success": False, "message": "File not found!"})
        
        try:
            os.remove(path)
            return JsonResponse({"success": True, "message": f"{filename} deleted successfully!"})
        except PermissionError:
            return JsonResponse({"success": False, "message": "File is currently in use by another program."})
        except Exception as e:
            return JsonResponse({"success": False, "message": f"Error deleting file: {str(e)}"})
    
    return JsonResponse({"success": False, "message": "Invalid request method"})

# ========================================
# ALGORITHMS
# ========================================
def algorithms(request):
    from django.shortcuts import render, redirect
    from django.contrib import messages
    from .ml.ml_cache import load_ml_results

    # 🔥 ONLY LOAD ML RESULTS (NO TRAINING)
    results = load_ml_results()

    if results is None:
        messages.error(
            request,
            "ML results not found. Train ML locally once and deploy."
        )
        return redirect("upload_file")

    # ---------------- REGRESSION TABLE ----------------
    reg_data = {
        "XGBoost": [],
        "Random Forest": [],
        "Linear": [],
        "Decision Tree": []
    }

    for key, mae in results.get("regressors", {}).get("models", {}).items():
        if "_" not in key:
            continue
        part, algo = key.rsplit("_", 1)
        if algo in reg_data:
            best_algo = results.get("regressors", {}).get("best", {}).get(part, "")
            reg_data[algo].append({
                "part": part,
                "mae": round(mae, 3),
                "best": "Best" if best_algo == algo else ""
            })

    # ---------------- CLASSIFICATION TABLE ----------------
    clf_data = {
        "XGBoost": [],
        "Random Forest": [],
        "Logistic": [],
        "Decision Tree": []
    }

    for key, acc in results.get("classifiers", {}).get("models", {}).items():
        if "_" not in key:
            continue
        part, algo = key.rsplit("_", 1)
        if algo in clf_data:
            best_algo = results.get("classifiers", {}).get("best", {}).get(part, "")
            clf_data[algo].append({
                "part": part,
                "acc": round(acc, 2),
                "best": "Best" if best_algo == algo else ""
            })

    return render(request, "analysi/algorithms.html", {
        "reg_data": reg_data,
        "clf_data": clf_data
    })

# ========================================
# RAW DATA VIEW
# ========================================


def get_latest_file(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.xlsx')]
    if not files:
        return None
    files = sorted(files, key=lambda x: os.path.getmtime(os.path.join(directory, x)), reverse=True)
    return files[0]



def raw_data_view(request):

    upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
    os.makedirs(upload_dir, exist_ok=True)

    latest_file = get_latest_file(upload_dir)
    if not latest_file:
        return render(request, 'analysi/raw_data.html', {
            'error': 'No valid Excel file found. Please upload BOX SCORE.xlsx first.'
        })

    file_path = os.path.join(upload_dir, latest_file)

    # Load raw data without headers for table display
    try:
        df_raw = pd.read_excel(file_path, sheet_name='BOX SCORE', header=None, engine='openpyxl')
    except Exception as e:
        return render(request, 'analysi/raw_data.html', {
            'error': f'Error reading file: {str(e)}'
        })

    raw_data = df_raw.fillna('').values.tolist()
    split_idx = 0
    for i, row in enumerate(raw_data):
        if any(str(c).strip().upper() == "CATEGORY" for c in row):
            split_idx = i
            break

    main_table = raw_data[:split_idx] + raw_data[split_idx:]
    main_table_cols_range = list(range(len(main_table[0]))) if main_table else []

    # Prepare chart data using your extract_data function
    try:
        df_clean = extract_data(file_path)
        
        # Extract dates and failures from cleaned data
        chart_dates = df_clean['date'].dt.strftime('%Y-%m-%d').tolist()
        chart_failures = df_clean['failure'].tolist()

        print("Chart Dates Sample:", chart_dates[:10])
        print("Chart Failures Sample:", chart_failures[:10])
        print("Total chart data points:", len(chart_dates))

    except Exception as e:
        print("Error preparing chart data:", str(e))
        chart_dates = []
        chart_failures = []

    return render(request, 'analysi/raw_data.html', {
        'file_name': latest_file,
        'main_table': main_table,
        'main_table_cols_range': main_table_cols_range,
        'chart_dates': chart_dates,
        'chart_failures': chart_failures,
    })


# ========================================
# FUTURE PREDICTIONS
# ========================================
# ========================================
# FUTURE PREDICTIONS - FIXED VERSION
# ========================================
# ========================================
# FUTURE PREDICTIONS - ENHANCED WITH FINANCIAL DATA
# ========================================


def get_latest_file(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.xlsx')]
    if not files:
        return None
    files = sorted(files, key=lambda x: os.path.getmtime(os.path.join(directory, x)), reverse=True)
    return files[0]


def get_latest_file(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.xlsx')]
    if not files:
        return None
    files = sorted(files, key=lambda x: os.path.getmtime(os.path.join(directory, x)), reverse=True)
    return files[0]


def future_predictions(request):
    import json
    from collections import Counter
    from .ml.ml_cache import load_ml_results

    results = load_ml_results()
    if not results:
        messages.error(request, "ML results not found. Run Algorithms once.")
        return redirect("algorithms")

    predictions = results.get("next_month_30_predictions", [])

    # 🔥 THIS WAS THE PROBLEM
    if not predictions:
        messages.error(request, "Future 30-day predictions not generated.")
        return render(request, "analysi/future_predictions.html", {
            "thirty_day": [],
            "failures_days": [],
            "chart_datasets_json": "[]",
            "reason_table": {},
            "latest_file": "",
        })

    for p in predictions:
        p["part"] = clean_category_name(p.get("part", ""))

    categories = ["FEED PUMP", "WATER PUMP", "PCN", "COOLANT ELBOW & COVERS", "PULLEY"]
    all_days = sorted(set(p["day"] for p in predictions), key=lambda x: int(x.split(" ")[1]))

    chart_datasets = []
    for cat in categories:
        cat_preds = [p for p in predictions if p["part"] == cat]
        data = [next((p["failure"] for p in cat_preds if p["day"] == d), 0) for d in all_days]

        color = get_category_color(cat)
        chart_datasets.append({
            "label": cat,
            "data": data,
            "borderColor": color["border"],
            "backgroundColor": color["background"],
            "fill": False,
            "tension": 0.4,
        })

    # Reason table
    reason_table = {}
    for p in predictions:
        cat = p.get("part")
        reason = p.get("reason")
        actual = p.get("predicted_actual", 0)

        if cat and reason:
            reason_table.setdefault(cat, {})
            reason_table[cat][reason] = reason_table[cat].get(reason, 0) + actual

    reason_table = {
        cat: [{"reason": r, "future_actual": v} for r, v in reasons.items()]
        for cat, reasons in reason_table.items()
    }

    return render(request, "analysi/future_predictions.html", {
        "latest_file": results.get("latest_file", ""),
        "thirty_day": predictions,
        "month_name": results.get("future_month_name", ""),
        "failures_days": all_days,
        "chart_datasets_json": json.dumps(chart_datasets),
        "reason_table": reason_table,
        "seven_day": results.get("predictions", []),
    })

# Helper function for colors (add this if not in your views.py)
def get_category_color(category):
    colors = {
        "FEED PUMP": {"border": "rgb(220, 38, 38)", "background": "rgba(220, 38, 38, 0.1)"},
        "WATER PUMP": {"border": "rgb(37, 99, 235)", "background": "rgba(37, 99, 235, 0.1)"},
        "PCN": {"border": "rgb(5, 150, 105)", "background": "rgba(5, 150, 105, 0.1)"},
        "COOLANT ELBOW & COVERS": {"border": "rgb(168, 85, 247)", "background": "rgba(168, 85, 247, 0.1)"},
        "PULLEY": {"border": "rgb(245, 158, 11)", "background": "rgba(245, 158, 11, 0.1)"},
    }
    return colors.get(category, {"border": "rgb(75, 85, 99)", "background": "rgba(75, 85, 99, 0.1)"})

# Helper function to clean category names (add this if not in your views.py)
def clean_category_name(name):
    # Remove any unwanted characters and standardize
    name = name.strip().upper()
    # Add your cleaning logic here if needed
    return name
# ========================================
# FINANCIAL DASHBOARD
# ========================================

def financial_dashboard(request):
  
    upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
    os.makedirs(upload_dir, exist_ok=True)

    latest_file = get_latest_file(upload_dir)
    if not latest_file:
        messages.error(request, "No Excel file uploaded!")
        return redirect('upload_file')

    file_path = os.path.join(upload_dir, latest_file)

    df_raw = pd.read_excel(file_path, nrows=800)
    if df.empty:
        messages.error(request, "No data extracted!")
        return redirect('upload_file')

    try:
        df_raw = pd.read_excel(file_path,nrows=800)
    except Exception as e:
        print(f"Error reading raw Excel: {e}")
        df_raw = None

    results = train_models(df, df_raw)

    if not isinstance(results, dict):
        messages.error(request, "Model training failed!")
        return redirect('upload_file')

    # Extract financial data
    financial_summary = results.get("financial_summary", {})
    financial_economics = results.get("financial_economics", {})
    
    # Prepare financial charts data
    category_revenues = {}
    category_profits = {}
    
    for pred in results.get("predictions", []) + results.get("next_month_30_predictions", []):
        category = pred["part"]
        financial = pred.get("financial_metrics", {})
        
        if category not in category_revenues:
            category_revenues[category] = 0
            category_profits[category] = 0
            
        category_revenues[category] += financial.get("planned_revenue", 0)
        category_profits[category] += financial.get("planned_gross_profit", 0)
    
    context = {
        "latest_file": latest_file,
        "financial_summary": financial_summary,
        "financial_economics": financial_economics,
        "category_revenues": category_revenues,
        "category_profits": category_profits,
        "investment_opportunities": financial_summary.get("investment_opportunities", []),
        "category_performance": financial_summary.get("category_performance", {}),
    }
    
    return render(request, 'analysi/financial_dashboard.html', context)
# ========================================
# DELETE PAGE LIST VIEW
# ========================================
# ========================================
# DELETE PAGE LIST VIEW - IMPROVED
# ========================================
def delete_page(request):
    upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    
    files = []
    for f in os.listdir(upload_dir):
        if f.endswith(('.xlsx', '.xls')) and not f.startswith('~$'):
            file_path = os.path.join(upload_dir, f)
            file_info = {
                'name': f,
                'size': os.path.getsize(file_path),
                'upload_time': datetime.fromtimestamp(os.path.getctime(file_path))
            }
            files.append(file_info)
    
    # Sort by upload time (newest first)
    files.sort(key=lambda x: x['upload_time'], reverse=True)
    
    return render(request, 'analysi/delete_page.html', {'files': files})