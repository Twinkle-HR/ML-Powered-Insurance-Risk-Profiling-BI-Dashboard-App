from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
from predict_helper import preprocess_input,compute_age, get_age_group, categorize_income, region_map

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import json
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from datetime import datetime

app = Flask(__name__)

# Load the trained model and label encoders
with open(r'C:\Users\Devashree\Downloads\DBDA_Project_Insurance\models\catboost_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load encoders if needed
with open(r'C:\Users\Devashree\Downloads\DBDA_Project_Insurance\models\label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

# In Flask app:
with open(r'C:\Users\Devashree\Downloads\DBDA_Project_Insurance\models\model_columns.pkl', 'rb') as f:
    final_columns = pickle.load(f)
    print("Model expects:", final_columns)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/classification")
def classification():
    return render_template("classify.html")

@app.route("/predict", methods=["POST"])
def predict_single():
    form = request.form

# 1) raw DOB → compute age
    dob = form['dob']
    age = compute_age(dob)

    # 2) derive age_group via helper
    age_group = get_age_group(age)

    annual_income = int(form['annual_income'])
    monthly_income = round(annual_income / 12)

    prods = form.getlist("insurance_products[]")
    ins_str = ",".join(prods)
    def has(label): return 1 if label in prods else 0

    customer_type = form['customer_type'] 

    input_dict = {
        'age': age,
        'age_group': age_group,        
        'gender': form['gender'],
        'marital_status': form['marital_status'],
        'education_level': form['education_level'],  
        'income_level': monthly_income,  
        'occupation_group': form['occupation'],  
        'credit_score': int(form['credit_score']),
        'state': form['state'],                  
        'policy_1': has("Health"),
        'policy_2': has("Home"),
        'policy_3': has("Vehicle"),
        'policy_4': has("Life/General"),
        'policy_5': has("Travel"),
        'policy_type': form['policy_type'],
        'policy_purchase_year': int(form['policy_purchase_year']),
        'coverage_amount': int(form['coverage_amount']),
        'premium_amount': int(form['premium_amount']),
        'deductible': int(form['deductible']),
       'previous_claims_history': 0 if customer_type=="new" else int(form['previous_claims_history']),
        'claim_history':          0 if customer_type=="new" else int(form['claim_history']),
        'driving_record': int(form['driving_record']),
        'life_events':   "No event" if customer_type=="new" else form['life_events'],
        'customer_preferences': form['customer_preferences'],
        'customer_type': customer_type
    }

    # 2) Turn into df, derive all other needed cols
    df = pd.DataFrame([input_dict])
    df = preprocess_input(df)  

    # 3) Encode & select exactly the model’s features
    for col, le in label_encoders.items():
        if col in df:
            df[col] = df[col].apply(lambda v: v if v in le.classes_ else le.classes_[0])
            df[col] = le.transform(df[col])

    df = df[[
        'age_group','gender','marital_status','education_category','occupation_group',
        'income_group','credit_score','state','region','policy_1','policy_2',
        'policy_3','policy_4','policy_5','policy_type','policy_purchase_year',
        'coverage_amount','premium_amount','deductible','premium_to_coverage_ratio',
        'previous_claims_history','claim_history',
        'driving_record','life_events','customer_preferences','customer_type'
    ]]

# 4) Predict
    pred = int(model.predict(df).flatten()[0])    
    conf = float(model.predict_proba(df).max()) * 100
    labels = {0: "Low", 1: "Medium", 2: "High", 3: "Very High"}

    return render_template(
        "classify.html",
        prediction=labels[pred],
        confidence=f"{conf:.1f}"
    )

# 1. Mount Dash on your Flask server

dash_app = dash.Dash(
    __name__,
    server=app,
    url_base_pathname="/kpi/dashboard/",
    external_stylesheets=["/static/css/kpi.css"]
)
# Load the KPI CSV once
df_full = pd.read_csv("data/kpi_data.csv")

# 2. Build Dash layout

dash_app.layout = html.Div(
    className="dashboard-right", 
    children=[
            html.Div(
            style={"display": "flex", "gap": "1rem", "marginBottom": "1rem"},
            children=[
                html.Div([
                html.Label("Year"),
                dcc.Dropdown(
                       id="year-dropdown",
                       options=[
                           {"label": "All",   "value": "all"},
                           *[
                             {"label": str(year), "value": str(year)}
                             for year in sorted(df_full["policy_purchase_year"].unique())
                           ]
                       ],
                       value="all", 
                       clearable=False
                   )
            ], style={"flex": "1"})
            ,
                html.Div([
                    html.Label("Age Range"),
                    dcc.RangeSlider(
                        id="age-slider",
                        min=int(df_full["age"].min()),
                        max=int(df_full["age"].max()),
                        value=[
                            int(df_full["age"].min()),
                            int(df_full["age"].max())
                        ],
                        marks={i: str(i) for i in range(
                            int(df_full["age"].min()),
                            int(df_full["age"].max()) + 1,
                            5
                        )},
                        step=1
                    )
                ], style={"flex": "3"})
            ]
        ),
        html.Div(
            className="chart-grid",
            children=[
                html.Div(dcc.Graph(id="policy-chart"),  className="chart-box"),
                html.Div(dcc.Graph(id="gender-chart"),  className="chart-box"),
                html.Div(dcc.Graph(id="location-chart"),className="chart-box"),
                html.Div(dcc.Graph(id="age-chart"),     className="chart-box"),
                html.Div(dcc.Graph(id="claims-chart"),  className="chart-box"),
                html.Div(dcc.Graph(id="risk-chart"),    className="chart-box"),
            ]
        )
    ]
)

@dash_app.callback(
    Output("policy-chart",   "figure"),
    Output("gender-chart",   "figure"),
    Output("location-chart", "figure"),
    Output("age-chart",      "figure"),
    Output("claims-chart",   "figure"),
    Output("risk-chart",     "figure"),
    Input("year-dropdown", "value"),
    Input("age-slider",    "value")
)

def update_kpis(selected_year, age_range):
    df = df_full.copy()
    low, high = age_range
    df = df[df["age"].between(low, high)]
    if selected_year != "all":
        df = df[df["policy_purchase_year"] == int(selected_year)]

    # map numeric to descriptive risk
    risk_map = {0: "No Risk", 1: "Low Risk", 2: "Medium Risk", 3: "High Risk"}
    df["risk_label"] = df["risk_profile"].map(risk_map)

    # 1) Risk Profile Distribution
    rp = (
        df["risk_label"]
          .value_counts()
          .reindex(list(risk_map.values()), fill_value=0)
          .rename_axis("Risk Level")
          .reset_index(name="Count")
    )
    fig1 = px.bar(rp, x="Risk Level", y="Count", title="Risk Profile Distribution")

    # 2) Risk Profile by Region (stacked horizontal)
    fig2 = px.histogram(
        df, y="region", color="risk_label",
        orientation="h", barmode="stack",
        category_orders={"risk_label": list(risk_map.values())},
        labels={"region":"Region","count":"Count","risk_label":"Risk Level"},
        title="Risk Profile by Region"
    )

    # 3) Policy Distribution (sum of policy_1…policy_5)
    policy_map = {
        "policy_1": "Health",
        "policy_2": "Home",
        "policy_3": "Vehicle",
        "policy_4": "Life",
        "policy_5": "Travel"
    }
    policy_counts = [
        {"Policy": policy_map[col], "Count": int(df[col].sum())}
        for col in policy_map
    ]
    pdistr = pd.DataFrame(policy_counts)
    fig3 = px.bar(
        pdistr,
        x="Policy",
        y="Count",
        title="Policy Distribution"
    )

    # 4) Income Category vs Risk Profile
    fig4 = px.histogram(
        df, x="income_group", color="risk_label",
        barmode="stack",
        category_orders={"risk_label": list(risk_map.values())},
        labels={"income_group":"Income Group","count":"Count","risk_label":"Risk Level"},
        title="Income Category vs Risk Profile"
    )

    # 5) Policy Type vs Risk Profile
    fig5 = px.histogram(
        df, x="policy_type", color="risk_label",
        barmode="group",
        category_orders={"risk_label": list(risk_map.values())},
        labels={"policy_type":"Policy Type","count":"Count","risk_label":"Risk Level"},
        title="Policy Type vs Risk Profile"
    )

    # 6) Age Distribution by Gender
    fig6 = px.histogram(
        df, x="age", color="gender",
        nbins=20, barmode="overlay",
        labels={"age":"Age","count":"Count","gender":"Gender"},
        title="Age Distribution by Gender"
    )

    for fig in (fig1, fig2, fig3, fig4, fig5, fig6):
        fig.update_layout(
            height=240,
            margin=dict(l=10, r=10, t=40, b=30),
            title_font=dict(size=14),
            font=dict(size=12),
            xaxis=dict(tickfont=dict(size=10)),
            yaxis=dict(title_font=dict(size=12), tickfont=dict(size=10))
        )

    return fig1, fig2, fig3, fig4, fig5, fig6

# ——————————————————————————————————————————
# 2) FLASK /kpi ROUTE with same risk_map
# ——————————————————————————————————————————
@app.route("/kpi")
def kpi_reporting():
    df = pd.read_csv("data/kpi_data.csv")

    total_claims    = df["claim_history"].sum()
    total_customers = df["customer_id"].nunique()
    total_premium   = df["premium_amount"].sum()
    avg_claims      = df.groupby("customer_id")["claim_history"].sum().mean() * 100
    high_risk_pct   = df["risk_profile"].eq(3).mean() * 100

    risk_map = {0: "No Risk", 1: "Low Risk", 2: "Medium Risk", 3: "High Risk"}
    df["risk_label"] = df["risk_profile"].map(risk_map)

    def make_figures(d):
        figs = []

        # 1) Risk Profile Distribution
        rp = (
            d["risk_label"]
             .value_counts()
             .reindex(list(risk_map.values()), fill_value=0)
             .rename_axis("Risk Level")
             .reset_index(name="Count")
        )
        figs.append(px.bar(rp, x="Risk Level", y="Count", title="Risk Profile Distribution"))

        # 2) Risk Profile by Region
        figs.append(px.histogram(
            d, y="region", color="risk_label",
            orientation="h", barmode="stack",
            category_orders={"risk_label": list(risk_map.values())},
            labels={"region":"Region","count":"Count","risk_label":"Risk Level"},
            title="Risk Profile by Region"
        ))

        # 3) Policy Distribution
        policy_map = {
            "policy_1": "Health",
            "policy_2": "Home",
            "policy_3": "Vehicle",
            "policy_4": "Life",
            "policy_5": "Travel"
        }
        policy_counts = [
            {"Policy": policy_map[col], "Count": int(d[col].sum())}
            for col in policy_map
        ]
        pdistr = pd.DataFrame(policy_counts)
        figs.append(px.bar(pdistr, x="Policy", y="Count", title="Policy Distribution"))

        # 4) Income Category vs Risk Profile
        figs.append(px.histogram(
            d, x="income_group", color="risk_label",
            barmode="stack",
            category_orders={"risk_label": list(risk_map.values())},
            labels={"income_group":"Income Group","count":"Count","risk_label":"Risk Level"},
            title="Income Category vs Risk Profile"
        ))

        # 5) Policy Type vs Risk Profile
        figs.append(px.histogram(
            d, x="policy_type", color="risk_label",
            barmode="group",
            category_orders={"risk_label": list(risk_map.values())},
            labels={"policy_type":"Policy Type","count":"Count","risk_label":"Risk Level"},
            title="Policy Type vs Risk Profile"
        ))

        # 6) Age Distribution by Gender
        figs.append(px.histogram(
            d, x="age", color="gender",
            nbins=20, barmode="overlay",
            labels={"age":"Age","count":"Count","gender":"Gender"},
            title="Age Distribution by Gender"
        ))

        out = []
        for f in figs:
            f.update_layout(height=240, margin=dict(l=10, r=10, t=40, b=30))
            js = json.loads(f.to_json())
            out.append({"data": js["data"], "layout": js["layout"]})
        return out

    charts = make_figures(df)

    return render_template(
        "kpi.html",
        total_claims=f"{total_claims:,}",
        total_customers=f"{total_customers:,}",
        total_premium=f"₹{total_premium:,.0f}",
        avg_claims=f"{avg_claims:.1f}",
        high_risk_percent=f"{high_risk_pct:.1f}",
        all_plots=charts
    )

if __name__ == "__main__":
    app.run(debug=True)
    
