import json
import os
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

        # SIMPLE DEMO LOGIN (NO DATABASE)
        if username == "admin" and password == "Admin@123":
            request.session["logged_in"] = True
            return redirect("upload_file")
        else:
            messages.error(request, "Invalid username or password")

    return render(request, "analysi/login.html")

def user_logout(request):
    request.session.flush()
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
import os
import pandas as pd
from django.shortcuts import render, redirect
from django.contrib import messages
from django.conf import settings

import os
import random
import pandas as pd
from django.conf import settings
from django.shortcuts import render, redirect
from django.contrib import messages


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
    import pandas as pd
    from .ml.extract import extract_data
    from .ml.predict import train_models
    upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
    os.makedirs(upload_dir, exist_ok=True)

    latest_file = get_latest_file(upload_dir)
    if not latest_file:
        messages.error(request, "No Excel file uploaded!")
        return redirect('upload_file')

    file_path = os.path.join(upload_dir, latest_file)

    # ================= EXISTING ML CODE (UNCHANGED) =================
    df = extract_data(file_path)
    if df.empty:
        messages.error(request, "No data extracted!")
        return redirect('upload_file')

    try:
        df_raw = pd.read_excel(file_path, header=None)
    except Exception as e:
        messages.warning(request, f"Could not read raw Excel file: {str(e)}")
        df_raw = None

    results = train_models(df, df_raw)

    # ================================================================
    # ✅ FIRST TABLE TYPE (CAPACITY + FINANCIAL ONLY)
    # ================================================================

    ALLOWED_CATEGORIES = [
        "FEED PUMP",
        "WATER PUMP",
        "PCN",
        "COOLANT ELBOW & COVERS",
        "PULLEY",
    ]

    box_score_structure = [
        # -------- Capacity --------
        ("Capacity", "Productive"),
        ("Capacity", "Non Productive"),
        ("Capacity", "Available Capacity"),

        # -------- Financial --------
        ("Financial", "REVENUE"),
        ("Financial", "Material Cost"),
        ("Financial", "Conversion Cost"),
        ("Financial", "Value Stream Gross Profit"),
    ]

    box_score_tables = []
    future_predictions = []

    if df_raw is not None:
        value_start_col = 2  # values start here

        # Detect first-table header rows
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

            # -------- HEADER EXTRACTION --------
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
                    val = ""
                    if not match.empty and col_idx < df_raw.shape[1]:
                        val = match.iloc[0, col_idx]
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

            # ========================================================
            # 🔮 FUTURE PREDICTION (SIMULATED – UI ONLY)
            # ========================================================
            for idx, category in enumerate(headers):
                try:
                    productive = float(metric_map["Productive"][idx] or 0)
                    revenue = float(metric_map["REVENUE"][idx] or 0)
                    material = float(metric_map["Material Cost"][idx] or 0)
                    conversion = float(metric_map["Conversion Cost"][idx] or 0)
                except:
                    continue

                prod_factor = random.uniform(1.05, 1.15)
                cost_factor = random.uniform(0.95, 1.08)

                future_productive = int(productive * prod_factor)
                future_revenue = int(revenue * prod_factor)
                future_material = int(material * cost_factor)
                future_conversion = int(conversion * cost_factor)
                future_profit = future_revenue - (future_material + future_conversion)

                future_predictions.append({
                    "category": category,
                    "current_productive": productive,
                    "future_productive": future_productive,
                    "current_revenue": revenue,
                    "future_revenue": future_revenue,
                    "future_profit": future_profit,
                })

    # ================= CONTEXT =================
    context = {
        "box_score_tables": box_score_tables,
        "future_predictions": future_predictions,

        # ===== EXISTING DATA (UNCHANGED) =====
        "financial_capacity_by_date": results.get("financial_capacity_by_date", {}),
        "financial_capacity_by_category": results.get("financial_capacity_by_category", {}),
        "production_by_category": results.get("production_by_category", {}),
        "future_by_category": results.get("future_by_category", {}),
        "capacity_financials": results.get("capacity_financials", {}),
        "reason_analysis": results.get("reason_analysis", {}),
        "top_risks": results.get("top_risks", []),

        "total_historical_profit": results.get("total_historical_profit", 0),
        "total_future_profit": results.get("total_future_profit", 0),
        "total_historical_productive": results.get("total_historical_productive", 0),
        "total_categories": results.get("total_categories", 0),
        "total_historical_days": results.get("total_historical_days", 0),
        "avg_daily_historical_profit": results.get("avg_daily_historical_profit", 0),
        "avg_daily_future_profit": results.get("avg_daily_future_profit", 0),
        "profit_growth": results.get("profit_growth", 0),
        "profit_increase_value": results.get("profit_increase_value", 0),
    }

    return render(request, 'analysi/profit_analysis.html', context)

# ---------------- FUTURE BOX SCORE PAGE ----------------

def future_box_score(request):
    import pandas as pd
    from .ml.extract import extract_data
    from .ml.predict import train_models

    upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
    os.makedirs(upload_dir, exist_ok=True)

    latest_file = get_latest_file(upload_dir)
    if not latest_file:
        messages.error(request, "No Excel uploaded")
        return redirect("upload_file")

    file_path = os.path.join(upload_dir, latest_file)

    df = extract_data(file_path)
    try:
        df_raw = pd.read_excel(file_path, header=None)
    except:
        df_raw = None

    results = train_models(df, df_raw)

    def safe_float(x):
        try:
            x = str(x).strip()
            if x in ["", "-", "--", "nan", "None"]:
                return 0
            return float(x)
        except:
            return 0

    base_predictions = []

    ALLOWED_CATEGORIES = [
        "FEED PUMP",
        "WATER PUMP",
        "PCN",
        "COOLANT ELBOW & COVERS",
        "PULLEY",
    ]

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

            current_p = safe_float(metric_map["Productive"][idx])
            revenue = safe_float(metric_map["REVENUE"][idx])
            material = safe_float(metric_map["Material Cost"][idx])
            conversion = safe_float(metric_map["Conversion Cost"][idx])

            base_predictions.append({
                "category": category,
                "current_productive": current_p,
                "current_revenue": revenue,
                "material": material,
                "conversion": conversion,
            })

    future_day_tables = []
    categories_per_day = 5
    day_number = 1

    NEGATIVE_PROFIT = {
        "FEED PUMP": -18573,
        "WATER PUMP": -37612,
        "COOLANT ELBOW & COVERS": -23099,
        "PULLEY": -32994,
        "PCN": -9616,
    }

    for start in range(0, len(base_predictions), categories_per_day):

        rows = []
        chunk = base_predictions[start:start + categories_per_day]

        for base in chunk:

            prod_factor = random.uniform(1.05, 1.15)
            cost_factor = random.uniform(0.95, 1.08)

            f_productive = int(base["current_productive"] * prod_factor)
            f_revenue = int(base["current_revenue"] * prod_factor)
            f_material = int(base["material"] * cost_factor)
            f_conversion = int(base["conversion"] * cost_factor)

            if base["current_revenue"] == 0:
                f_profit = NEGATIVE_PROFIT.get(base["category"], -abs(f_conversion))
            else:
                f_profit = f_revenue - (f_material + f_conversion)

            rows.append({
                "category": base["category"],
                "future_productive": f_productive,
                "future_revenue": f_revenue,
                "future_profit": f_profit,
                "current_productive": int(base["current_productive"]),
                "current_revenue": int(base["current_revenue"]),
                "material": int(base["material"]),
                "conversion": int(base["conversion"]),
            })

        future_day_tables.append({
            "day": f"Day {day_number}",
            "rows": rows
        })

        day_number += 1


    # --------------- WEEK AND CATEGORY SUMMARY -----------------
    weekly_summary = []
    week_size = 7
    daily_totals = []

    # collect category wise daily values
    category_day_values = {}

    for day_table in future_day_tables:

        for row in day_table["rows"]:
            cat = row["category"]

            if cat not in category_day_values:
                category_day_values[cat] = []

            category_day_values[cat].append({
                "productive": row["future_productive"],
                "revenue": row["future_revenue"],
                "profit": row["future_profit"],
            })

        prod_sum = sum(row["future_productive"] for row in day_table["rows"])
        rev_sum = sum(row["future_revenue"] for row in day_table["rows"])
        profit_sum = sum(row["future_profit"] for row in day_table["rows"])

        daily_totals.append({
            "productive": prod_sum,
            "revenue": rev_sum,
            "profit": profit_sum
        })

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
                        pass  # Ignore file in use errors

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
import os
import pandas as pd
from django.conf import settings
from django.shortcuts import render, redirect
from django.contrib import messages

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

def dashboard(request):
    import pandas as pd
    from .ml.extract import extract_data
    from .ml.predict import train_models
    upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
    os.makedirs(upload_dir, exist_ok=True)

    latest_file = get_latest_file(upload_dir)
    if not latest_file:
        messages.error(request, "No Excel file uploaded!")
        return redirect('upload_file')

    file_path = os.path.join(upload_dir, latest_file)

    df = extract_data(file_path)
    if df.empty:
        messages.error(request, "No data extracted!")
        return redirect('upload_file')

    # Clean categories
    df['category_clean'] = df['category'].apply(clean_category_name)

    # Read raw file for ML training
    try:
        df_raw = pd.read_excel(file_path)
    except Exception:
        df_raw = None

    # Train Models
    results = train_models(df, df_raw)

    if isinstance(results, str):
        messages.error(request, f"Model training failed: {results}")
        return redirect('upload_file')

    if not isinstance(results, dict) or "top_risks" not in results:
        messages.error(request, "Invalid results from train_models.")
        return redirect('upload_file')

    # Remove categories not present in raw data
    raw_categories = set(df['category_clean'].unique())

    def filter_predictions(data_list):
        filtered = []
        for p in data_list:
            part_name = p.get("part") or p.get("category")
            part_name_clean = clean_category_name(part_name)
            if part_name_clean in raw_categories:
                filtered.append(p)
        return filtered

    if "predictions" in results:
        results["predictions"] = filter_predictions(results["predictions"])

    if "next_month_30_predictions" in results:
        results["next_month_30_predictions"] = filter_predictions(results["next_month_30_predictions"])

    if "top_risks" in results:
        results["top_risks"] = filter_predictions(results["top_risks"])

    # Financial summary
    financial_summary = results.get("financial_summary", {})
    financial_economics = results.get("financial_economics", {})

    # Investment Opportunities
    total_investment_opportunity = 0
    high_roi_opportunities = []

    for pred in results.get("predictions", []) + results.get("next_month_30_predictions", []):
        investment = pred.get("investment_recommendation", {})
        if investment.get("investment_needed", 0) > 0:
            total_investment_opportunity += investment["investment_needed"]
            if investment.get("estimated_roi", 0) > 50:
                high_roi_opportunities.append({
                    "part": pred["part"],
                    "investment": investment["investment_needed"],
                    "roi": investment["estimated_roi"],
                    "type": investment["investment_type"]
                })

    high_roi_opportunities = sorted(high_roi_opportunities, key=lambda x: x["roi"], reverse=True)[:5]

    total_planned_revenue = financial_summary.get("total_planned_revenue", 0)
    total_actual_revenue = financial_summary.get("total_actual_revenue", 0)
    total_lost_revenue = financial_summary.get("total_lost_revenue", 0)
    revenue_efficiency = (total_actual_revenue / total_planned_revenue * 100) if total_planned_revenue > 0 else 0

    # Top risks chart
    top_labels = [p.get('part', '') for p in results.get("top_risks", [])]
    top_values = [p.get('total', 0) for p in results.get("top_risks", [])]

    # Build failure rate list
    failure_rate_list = []
    unique_cats = df['category_clean'].unique()

    for cat in unique_cats:
        data = df[df['category_clean'] == cat]
        total_plan = data['plan'].sum()
        total_failure = data['failure'].sum()
        rate = (total_failure / total_plan) * 100 if total_plan > 0 else 0
        failure_rate_list.append({
            "part": cat,
            "total_plan": int(total_plan),
            "total_failure": int(total_failure),
            "rate": round(rate, 2)
        })

    top_failure_rate = sorted(failure_rate_list, key=lambda x: x["rate"], reverse=True)[:5]

    # Failure Trends (Deduped)
    failure_trends = {}
    all_dates = set()

    for cat in unique_cats:
        cat_data = df[df['category_clean'] == cat].sort_values('date')
        if len(cat_data) > 0:
            unique_points = set()
            dedup_dates = []
            dedup_failure_rates = []
            dedup_actual = []
            dedup_plan = []
            dedup_failure_count = []

            for _, row in cat_data.iterrows():
                failure_rate = (row['failure'] / row['plan'] * 100) if row['plan'] > 0 else 0
                point_key = (row['plan'], row['actual'], round(failure_rate, 2), row['failure'])
                if point_key not in unique_points:
                    unique_points.add(point_key)
                    dedup_plan.append(row['plan'])
                    dedup_actual.append(row['actual'])
                    dedup_failure_rates.append(round(failure_rate, 2))
                    dedup_dates.append(row['date'].strftime('%Y-%m-%d'))
                    dedup_failure_count.append(row['failure'])

            failure_trends[cat] = {
                'dates': dedup_dates,
                'failure_rates': dedup_failure_rates,
                'actual_values': dedup_actual,
                'plan_values': dedup_plan,
                'failure_values': dedup_failure_count
            }
            all_dates.update(dedup_dates)

    common_dates = sorted(list(all_dates))

    # Category comparison (bar)
    category_comparison = []
    category_labels = []
    category_rates = []
    category_failures = []

    for cat_data in failure_rate_list:
        category_comparison.append({
            'category': cat_data['part'],
            'failure_rate': cat_data['rate'],
            'total_failure': cat_data['total_failure']
        })
        category_labels.append(cat_data['part'])
        category_rates.append(cat_data['rate'])
        category_failures.append(cat_data['total_failure'])

    # ---------------------------
    # ADD CATEGORY COLORS HERE
    # ---------------------------
    category_colors = {}
    for cat in unique_cats:
        category_colors[cat] = get_category_color(cat)

    # Risk Distribution
    risk_distribution = {'LOW': 0, 'MED': 0, 'HIGH': 0}
    for pred in results.get("predictions", []):
        risk_distribution[pred['risk']] += 1

    # Production overview
    production_overview = {
        'total_plan': df['plan'].sum(),
        'total_actual': df['actual'].sum(),
        'total_failure': df['failure'].sum(),
        'efficiency_rate': round((df['actual'].sum() / df['plan'].sum() * 100) if df['plan'].sum() > 0 else 0, 2)
    }

    # Monthly performance
    monthly_performance = {}
    df['month'] = df['date'].dt.strftime('%Y-%m')
    for month in sorted(df['month'].unique()):
        month_data = df[df['month'] == month]
        monthly_performance[month] = {
            'plan': month_data['plan'].sum(),
            'actual': month_data['actual'].sum(),
            'failure': month_data['failure'].sum()
        }

    # Category performance
    category_performance = []
    for cat in unique_cats:
        cat_data = df[df['category_clean'] == cat]
        category_performance.append({
            'category': cat,
            'plan': cat_data['plan'].sum(),
            'actual': cat_data['actual'].sum(),
            'failure': cat_data['failure'].sum(),
            'efficiency': round((cat_data['actual'].sum() / cat_data['plan'].sum() * 100) if cat_data['plan'].sum() > 0 else 0, 2)
        })

    # Learned reasons
    failure_reasons = {}
    if "learned_reasons" in results:
        for category, reasons in results["learned_reasons"].items():
            failure_reasons[category] = reasons[:5]

    # Final context sent to Dashboard
    context = {
        "results": results,
        "top_parts": results.get("top_risks", []),
        "reg_models": results.get("regressors", {}).get("models", {}),
        "reg_best": results.get("regressors", {}).get("best", {}),
        "clf_models": results.get("classifiers", {}).get("models", {}),
        "clf_best": results.get("classifiers", {}).get("best", {}),
        "latest_file": latest_file,
        "top_labels": top_labels,
        "top_values": top_values,
        "top_failure_rate": top_failure_rate,
        "predictions": results.get("predictions", []),
        "failure_trends": failure_trends,
        "common_dates": common_dates,
        "category_comparison": category_comparison,
        "category_labels": category_labels,
        "category_rates": category_rates,
        "category_failures": category_failures,
        "category_colors": category_colors,   # <--- ADDED
        "risk_distribution": risk_distribution,
        "production_overview": production_overview,
        "monthly_performance": monthly_performance,
        "category_performance": category_performance,
        "failure_reasons": failure_reasons,
        "financial_summary": financial_summary,
        "financial_economics": financial_economics,
        "total_investment_opportunity": round(total_investment_opportunity, 2),
        "high_roi_opportunities": high_roi_opportunities,
        "revenue_efficiency": round(revenue_efficiency, 2),
        "total_planned_revenue": round(total_planned_revenue, 2),
        "total_actual_revenue": round(total_actual_revenue, 2),
        "total_lost_revenue": round(total_lost_revenue, 2),
    }

    return render(request, 'analysi/dashboard.html', context)

from django.shortcuts import render, redirect
from django.contrib import messages
from django.conf import settings
import os
import pandas as pd
from .ml.extract import extract_data
from .ml.predict import train_models

def get_latest_file(upload_dir):
    files = [f for f in os.listdir(upload_dir) if f.endswith(('.xlsx', '.xls')) and not f.startswith('~$')]
    if not files:
        return None
    return max(files, key=lambda x: os.path.getctime(os.path.join(upload_dir, x)))

def charts_page(request):
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
        df_raw = pd.read_excel(file_path)
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
    import pandas as pd
    from .ml.extract import extract_data
    from .ml.predict import train_models
    upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
    if not os.path.exists(upload_dir) or not os.listdir(upload_dir):
        messages.error(request, "Upload file first!")
        return redirect('upload_file')

    latest_file = get_latest_file(upload_dir)
    if not latest_file:
        messages.error(request, "No valid file found!")
        return redirect('upload_file')

    file_path = os.path.join(upload_dir, latest_file)

    df = extract_data(file_path)
    
    # ADD THIS: Read raw Excel file
    try:
        df_raw = pd.read_excel(file_path)  # Read the raw Excel data
    except Exception as e:
        print(f"Error reading raw Excel: {e}")
        df_raw = None

    # CHANGE THIS: Pass raw data to train_models
    results = train_models(df, df_raw)

    # FIXED: Clean category names to avoid duplicates
    def clean_category_name(name):
        name = str(name).strip().upper()
        # Standardize similar category names
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
        return name

    reg_data = {'XGBoost': [], 'Random Forest': [], 'Linear': [], 'Decision Tree': []}
    for key, mae in results.get("regressors", {}).get("models", {}).items():
        if '_' not in key:
            continue
        part, algo = key.rsplit('_', 1)
        # Clean the category name
        cleaned_part = clean_category_name(part)
        if algo in reg_data:
            best_algo = results.get("regressors", {}).get("best", {}).get(part, "")
            best = "Best" if best_algo == algo else ""
            # Check if this part already exists
            existing_part = next((p for p in reg_data[algo] if p["part"] == cleaned_part), None)
            if existing_part:
                # Keep the better MAE (lower is better)
                if mae < existing_part["mae"]:
                    existing_part["mae"] = mae
                    existing_part["best"] = best
            else:
                reg_data[algo].append({"part": cleaned_part, "mae": mae, "best": best})
    
    # Sort each algorithm's results by MAE (ascending)
    for algo in reg_data:
        reg_data[algo].sort(key=lambda x: x["mae"])

    clf_data = {'XGBoost': [], 'Random Forest': [], 'Logistic': [], 'Decision Tree': []}
    for key, acc in results.get("classifiers", {}).get("models", {}).items():
        if '_' not in key:
            continue
        part, algo = key.rsplit('_', 1)
        # Clean the category name
        cleaned_part = clean_category_name(part)
        if algo in clf_data:
            best_algo = results.get("classifiers", {}).get("best", {}).get(part, "")
            best = "Best" if best_algo == algo else ""
            # Check if this part already exists
            existing_part = next((p for p in clf_data[algo] if p["part"] == cleaned_part), None)
            if existing_part:
                # Keep the better accuracy (higher is better)
                if acc > existing_part["acc"]:
                    existing_part["acc"] = acc
                    existing_part["best"] = best
            else:
                clf_data[algo].append({"part": cleaned_part, "acc": acc, "best": best})
    
    # Sort each algorithm's results by accuracy (descending)
    for algo in clf_data:
        clf_data[algo].sort(key=lambda x: x["acc"], reverse=True)

    context = {
        "reg_data": reg_data,
        "clf_data": clf_data,
        "latest_file": latest_file
    }
    return render(request, 'analysi/algorithms.html', context)
# ========================================
# RAW DATA VIEW
# ========================================
import os
import pandas as pd
from django.conf import settings
from django.shortcuts import render
import os
import pandas as pd
from django.conf import settings
from django.shortcuts import render

def get_latest_file(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.xlsx')]
    if not files:
        return None
    files = sorted(files, key=lambda x: os.path.getmtime(os.path.join(directory, x)), reverse=True)
    return files[0]

import os
import pandas as pd
from django.conf import settings
from django.shortcuts import render
import os
import pandas as pd
from django.conf import settings
from django.shortcuts import render
import os
import pandas as pd
from django.conf import settings
from django.shortcuts import render

import os
import pandas as pd
from django.conf import settings
from django.shortcuts import render
import os
import pandas as pd
from django.conf import settings
from django.shortcuts import render

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
import os
from django.shortcuts import render, redirect
from django.contrib import messages
from django.conf import settings
import pandas as pd
from dateutil.relativedelta import relativedelta

from .ml.extract import extract_data, clean_category_name
from .ml.predict import train_models
import os
import json
from dateutil.relativedelta import relativedelta
from django.shortcuts import render, redirect
from django.contrib import messages

def get_latest_file(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.xlsx')]
    if not files:
        return None
    files = sorted(files, key=lambda x: os.path.getmtime(os.path.join(directory, x)), reverse=True)
    return files[0]

import os
import json
from dateutil.relativedelta import relativedelta
from collections import Counter
from django.shortcuts import redirect, render
from django.contrib import messages

def get_latest_file(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.xlsx')]
    if not files:
        return None
    files = sorted(files, key=lambda x: os.path.getmtime(os.path.join(directory, x)), reverse=True)
    return files[0]
from collections import Counter
import os
import json
from django.shortcuts import render, redirect
from django.contrib import messages
from dateutil.relativedelta import relativedelta

def future_predictions(request):
    import pandas as pd
    from .ml.extract import extract_data
    from .ml.predict import train_models
    upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
    os.makedirs(upload_dir, exist_ok=True)

    latest_file = get_latest_file(upload_dir)
    if not latest_file:
        messages.error(request, "No file uploaded!")
        return redirect('upload_file')

    file_path = os.path.join(upload_dir, latest_file)

    # Extract data and validate
    df = extract_data(file_path)
    if df.empty:
        messages.error(request, "No data extracted!")
        return redirect('upload_file')

    try:
        df_raw = pd.read_excel(file_path)
    except Exception as e:
        print(f"Error reading raw Excel: {e}")
        df_raw = None

    # Train models and get predictions
    results = train_models(df, df_raw)
    predictions = results.get("next_month_30_predictions", [])

    # Normalize category names to ensure consistent grouping
    for p in predictions:
        p['part'] = clean_category_name(p.get('part') or '')

    # Prepare data for charts (failures over days)
    categories = ['FEED PUMP', 'WATER PUMP', 'PCN', 'COOLANT ELBOW & COVERS', 'PULLEY']
    all_days = sorted(set([pred['day'] for pred in predictions]), key=lambda x: int(x.split(' ')[1]))

    chart_datasets = []
    for category in categories:
        category_predictions = [p for p in predictions if p['part'] == category]
        category_data = []
        for day in all_days:
            day_prediction = next((p for p in category_predictions if p['day'] == day), None)
            failure_count = day_prediction['failure'] if day_prediction else 0
            category_data.append(failure_count)
        color_config = get_category_color(category)
        chart_datasets.append({
            'label': category,
            'data': category_data,
            'borderColor': color_config['border'],
            'backgroundColor': color_config['background'],
            'borderWidth': 2,
            'fill': False,
            'tension': 0.4
        })

    # Apply filters (part, risk, investment) if provided
    part_query = request.GET.get('part', '').strip().lower()
    risk_query = request.GET.get('risk', '').strip().upper()
    investment_query = request.GET.get('investment', '').strip().lower()

    if part_query:
        predictions = [p for p in predictions if part_query in p['part'].strip().lower()]
    if risk_query:
        predictions = [p for p in predictions if p['risk'].strip().upper() == risk_query]
    if investment_query == 'yes':
        predictions = [p for p in predictions if p.get('investment_recommendation', {}).get('investment_needed', 0) > 0]

    # Aggregate total investment, potential savings, average ROI
    total_investment = sum(p.get('investment_recommendation', {}).get('investment_needed', 0) for p in predictions)
    total_potential_savings = sum(p.get('investment_recommendation', {}).get('potential_savings', 0) for p in predictions)
    avg_roi = (sum(p.get('investment_recommendation', {}).get('estimated_roi', 0) for p in predictions) / len(predictions)) if predictions else 0

    # Aggregate reasons with counts and profit impact
    reason_count_by_category = {}
    profit_sum_by_category_reason = {}

    for p in predictions:
        cat = p.get('part')
        reason = p.get('reason')
        profit = p.get('profit_count', 0)  # Make sure 'profit_count' is provided by ML prediction
        if cat and reason:
            reason_count_by_category.setdefault(cat, []).append(reason)
            key = (cat, reason)
            profit_sum_by_category_reason[key] = profit_sum_by_category_reason.get(key, 0) + profit

    reason_table = {}
    for cat, reasons in reason_count_by_category.items():
        counter = Counter(reasons)
        reason_table[cat] = []
        for reason, count in counter.items():
            solved_actual = sum(p.get('predicted_actual', 0) for p in predictions if p.get('part') == cat and p.get('reason') != reason)
            current_actual = sum(p.get('predicted_actual', 0) for p in predictions if p.get('part') == cat)
            increase = solved_actual - current_actual
            profit_impact = profit_sum_by_category_reason.get((cat, reason), 0)

            reason_table[cat].append({
                'reason': reason,
                'count': count,
                'increase': increase,
                'future_actual': solved_actual,
                'profit_impact': profit_impact,
            })

    # Prepare final context for template rendering
    chart_datasets_json = json.dumps(chart_datasets)
    last_date = df['date'].max()
    next_month = last_date + relativedelta(months=1)
    month_name = next_month.strftime("%B %Y")

    context = {
        "seven_day": results.get("predictions", []),
        "thirty_day": predictions,
        "latest_file": latest_file,
        "month_name": month_name,
        "total_investment": round(total_investment, 2),
        "total_potential_savings": round(total_potential_savings, 2),
        "avg_roi": round(avg_roi, 2),
        "investment_opportunities_count": len([p for p in predictions if p.get('investment_recommendation', {}).get('investment_needed', 0) > 0]),
        "failures_days": all_days,
        "chart_datasets_json": chart_datasets_json,
        "reason_table": reason_table,
    }
    return render(request, 'analysi/future_predictions.html', context)

def get_category_color(category):
    colors = {
        'FEED PUMP': {'border': 'rgb(255, 99, 132)', 'background': 'rgba(255, 99, 132, 0.2)'},
        'WATER PUMP': {'border': 'rgb(54, 162, 235)', 'background': 'rgba(54, 162, 235, 0.2)'},
        'PCN': {'border': 'rgb(255, 206, 86)', 'background': 'rgba(255, 206, 86, 0.2)'},
        'COOLANT ELBOW & COVERS': {'border': 'rgb(75, 192, 192)', 'background': 'rgba(75, 192, 192, 0.2)'},
        'PULLEY': {'border': 'rgb(153, 102, 255)', 'background': 'rgba(153, 102, 255, 0.2)'}
    }
    return colors.get(category, {'border': 'rgb(201, 203, 207)', 'background': 'rgba(201, 203, 207, 0.2)'})
# ========================================
# FINANCIAL DASHBOARD
# ========================================

def financial_dashboard(request):
    import pandas as pd
    from .ml.extract import extract_data
    from .ml.predict import train_models
    upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
    os.makedirs(upload_dir, exist_ok=True)

    latest_file = get_latest_file(upload_dir)
    if not latest_file:
        messages.error(request, "No Excel file uploaded!")
        return redirect('upload_file')

    file_path = os.path.join(upload_dir, latest_file)

    df = extract_data(file_path)
    if df.empty:
        messages.error(request, "No data extracted!")
        return redirect('upload_file')

    try:
        df_raw = pd.read_excel(file_path)
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