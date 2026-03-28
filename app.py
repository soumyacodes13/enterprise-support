import streamlit as st
import pandas as pd
from openai import OpenAI
import os
from dotenv import load_dotenv
from charset_normalizer import detect
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Sales Analytics", layout="wide", initial_sidebar_state="collapsed")

st.title("Sales Analytics")

file = st.file_uploader("Upload CSV", type=["csv"])

if file is not None:
    raw = file.read()
    encoding = detect(raw)['encoding'] or 'utf-8'
    file.seek(0)
    df = pd.read_csv(file, encoding=encoding)
    df.rename(columns={'Sale_Date': 'Date'}, inplace=True)

    if 'Profit' in df.columns:
        df['Cost'] = df['Sales_Amount'] - df['Profit']
        df['Unit_Cost'] = (df['Cost'] / df['Quantity_Sold'].replace(0, 1)).round(2)
    else:
        df['Unit_Price'] = df['Sales_Amount'] / df['Quantity_Sold'].replace(0, 1)
        df['Unit_Cost'] = df['Unit_Price'] * 0.65
        df['Cost'] = df['Unit_Cost'] * df['Quantity_Sold']
        df['Profit'] = df['Sales_Amount'] - df['Cost']

    required = ["Sales_Amount", "Unit_Cost", "Quantity_Sold", "Region", "Product_Category"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {', '.join(missing)}")
        st.stop()

    for col in ["Sales_Amount", "Unit_Cost", "Quantity_Sold"]:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce')

    df["Revenue"] = df["Sales_Amount"]
    df["Cost"] = df["Unit_Cost"] * df["Quantity_Sold"]
    df["Profit"] = df["Revenue"] - df["Cost"]
    df["Profit_Margin"] = (df["Profit"] / df["Revenue"].replace(0, float('nan')) * 100).fillna(0)

    total_revenue = df["Revenue"].sum()
    total_cost = df["Cost"].sum()
    total_profit = df["Profit"].sum()
    avg_margin = df["Profit_Margin"].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Revenue", f"${total_revenue:,.0f}")
    col2.metric("Cost", f"${total_cost:,.0f}")
    col3.metric("Profit", f"${total_profit:,.0f}")
    col4.metric("Avg Margin", f"{avg_margin:.1f}%")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Analytics", "Issues", "Root Cause", "Recommendations", "AI Insights"])

    with tab1:
        region_data = df.groupby("Region").agg(Revenue=("Revenue","sum"), Profit=("Profit","sum")).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Revenue', x=region_data['Region'], y=region_data['Revenue']))
        fig.add_trace(go.Bar(name='Profit', x=region_data['Region'], y=region_data['Profit']))
        fig.update_layout(barmode='group', title='Revenue & Profit by Region')
        st.plotly_chart(fig, width='stretch')

        c1, c2 = st.columns(2)
        with c1:
            prod = df.groupby("Product_Category")['Profit'].sum()
            st.plotly_chart(px.pie(values=prod.values, names=prod.index, title='Profit by Category'), width='stretch')
        with c2:
            marg = df.groupby("Product_Category")['Profit_Margin'].mean().sort_values()
            st.plotly_chart(go.Figure(go.Bar(x=marg.values, y=marg.index, orientation='h')).update_layout(title='Avg Margin by Category'), width='stretch')

    with tab2:
        neg = df[df["Profit"] < 0]
        region_profit = df.groupby("Region")["Profit"].sum().sort_values()
        product_margin = df.groupby("Product_Category")["Profit_Margin"].mean().sort_values()

        if not neg.empty:
            st.error(f"{len(neg)} loss-making items — total loss: ${abs(neg['Profit'].sum()):,.0f}")
            with st.expander("View"):
                st.dataframe(neg[["Region","Product_Category","Revenue","Cost","Profit"]].head(20), width='stretch')
        if not region_profit.empty:
            st.warning(f"Worst region: {region_profit.index[0]} — ${region_profit.iloc[0]:,.0f}")
        if not product_margin.empty:
            st.warning(f"Lowest margin: {product_margin.index[0]} — {product_margin.iloc[0]:.1f}%")

        high_rev_low = df[(df["Revenue"] > df["Revenue"].quantile(0.75)) & (df["Profit_Margin"] < df["Profit_Margin"].quantile(0.25))]
        if not high_rev_low.empty:
            st.info(f"{len(high_rev_low)} high-revenue, low-margin items — avg margin: {high_rev_low['Profit_Margin'].mean():.1f}%")

    with tab3:
        worst_neg_region = neg.groupby("Region")["Profit"].sum().sort_values().index[0] if not neg.empty else None
        worst_region = region_profit.index[0] if not region_profit.empty else None
        worst_cat = product_margin.index[0] if not product_margin.empty else None

        if worst_neg_region:
            st.write(f"Losses concentrated in **{worst_neg_region}**")
        if worst_region:
            worst_prod = df[df["Region"]==worst_region].groupby("Product_Category")["Profit"].sum().sort_values().index[0]
            st.write(f"Worst product in {worst_region}: **{worst_prod}**")
        if worst_cat:
            worst_reg_for_cat = df[df["Product_Category"]==worst_cat].groupby("Region")["Profit_Margin"].mean().sort_values().index[0]
            st.write(f"Lowest margin for {worst_cat}: **{worst_reg_for_cat}** region")

    with tab4:
        recs = []
        if total_profit < 0: recs.append("Implement cost reduction — currently at a net loss.")
        if worst_neg_region: recs.append(f"Renegotiate supplier contracts in {worst_neg_region}.")
        if worst_region: recs.append(f"Run a targeted push in {worst_region}.")
        if not high_rev_low.empty: recs.append("Review pricing on high-volume, low-margin lines.")
        if worst_cat: recs.append(f"Audit or phase out {worst_cat}.")
        for i, r in enumerate(recs, 1):
            st.write(f"{i}. {r}")

    with tab5:
        if st.button("Run analysis"):
            with st.spinner("Generating..."):
                load_dotenv()
                context = f"Revenue: ${total_revenue:,.0f} | Cost: ${total_cost:,.0f} | Profit: ${total_profit:,.0f} | Margin: {avg_margin:.1f}% | Loss items: {len(neg)}"
                try:
                    client = OpenAI(base_url="https://models.github.ai/inference", api_key=os.getenv("GITHUB_TOKEN"))
                    resp = client.chat.completions.create(
                        model="Meta-Llama-3.1-8B-Instruct",
                        messages=[{"role": "user", "content": f"Senior business consultant. 5-8 sentence executive summary with recommendations.\n\n{context}"}],
                        temperature=0.7, max_tokens=500
                    )
                    st.write(resp.choices[0].message.content.strip())
                except Exception as e:
                    st.error(str(e))