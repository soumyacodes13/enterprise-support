import streamlit as st
import pandas as pd
from openai import OpenAI
import os
from dotenv import load_dotenv
from charset_normalizer import detect

# At top of script
st.markdown("""
<style>
div[data-testid="stTabs"] {
    width: 200% !important;
}
div[data-testid="column"] {
    width: 200% !important;
}
</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="Enterprise Decision Support System", layout="wide")

st.title("📊 Enterprise Analytics & Decision Support System")
st.subheader("Upload business data to unlock actionable insights and automated decisions.")

# Step 1: File Upload
file = st.file_uploader("Upload your sales data (CSV)", type=["csv"])

if file is not None:
    with st.spinner("Processing data..."):
        try:
            raw = file.read()
            result = detect(raw)
            encoding = result['encoding'] or 'utf-8'
            file.seek(0)  # reset file pointer

            df = pd.read_csv(file, encoding=encoding)
            # st.info(f"Detected encoding: {encoding}")

            # df = pd.read_csv(file)
            df.rename(columns={'Sale_Date':'Date'},inplace=True)
            # After df = pd.read_csv(...) and renaming

            if 'Profit' in df.columns:
                df['Cost'] = df['Sales_Amount'] - df['Profit']
                df['Unit_Cost'] = df['Cost'] / df['Quantity_Sold'].replace(0, 1)
                df['Unit_Cost'] = df['Unit_Cost'].round(2)
                st.success("Unit_Cost calculated accurately using Profit = Sales - Cost")
            else:
                # Fallback if no Profit column (rare in Superstore)
                assumed_margin = 0.35
                df['Unit_Price'] = df['Sales_Amount'] / df['Quantity_Sold'].replace(0, 1)
                df['Unit_Cost'] = df['Unit_Price'] * (1 - assumed_margin)
                df['Cost'] = df['Unit_Cost'] * df['Quantity_Sold']
                df['Profit'] = df['Sales_Amount'] - df['Cost']
                st.info(f"No Profit column → assumed {assumed_margin*100:.0f}% gross margin to estimate Unit_Cost")
            # Step 2: Validate Columns
            required_cols = ["Sales_Amount", "Unit_Cost", "Quantity_Sold", "Region", "Product_Category"]
            optional_cols = ["Date", "Supplier"]  # For advanced analysis
            missing_required = [col for col in required_cols if col not in df.columns]
            if missing_required:
                st.error(f"Missing required columns: {', '.join(missing_required)}")
                st.stop()

            # Handle Missing Values & Conversions
            numeric_cols = ["Sales_Amount", "Unit_Cost", "Quantity_Sold"]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            # Fix missing values
            df[numeric_cols] = df[numeric_cols].fillna(0)


            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
                if df["Date"].isna().any():
                    st.warning("Some dates could not be parsed")

            # Check optional columns
            if "Supplier" not in df.columns:
                st.info("Supplier column missing")
            if "Date" not in df.columns:
                st.info("Date column missing")

            # Derived Columns 
            df["Cost"] = df["Unit_Cost"] * df["Quantity_Sold"]
            df["Revenue"] = df["Sales_Amount"]
            df["Profit"] = df["Revenue"] - df["Cost"]
            df["Profit_Margin"] = (df["Profit"] / df["Revenue"].replace(0, float('nan'))) * 100
            df["Profit_Margin"] = df["Profit_Margin"].fillna(0)  # Fill NaN margins with 0 for averaging
            
            st.success("Data loaded,validated,and metrics calculated.")
            with st.expander("View full data preview"):
                st.dataframe(df)
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "🔍 Summary",
                "⚠️ Problems",
                "🕵️ Root Causes",
                "💡 Recommendations",
                "🤖 AI Summary"
            ])
            with tab1:
                # Insight Engine
                st.header("🔍 Performance Summary")
                total_revenue = df["Revenue"].sum()
                total_cost = df["Cost"].sum()
                total_profit = df["Profit"].sum()
                avg_margin = df["Profit_Margin"].mean()

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Revenue", f"${total_revenue:,.2f}")
                col2.metric("Total Cost", f"${total_cost:,.2f}")
                col3.metric("Total Profit", f"${total_profit:,.2f}", delta="Loss" if total_profit < 0 else "Gain")
                col4.metric("Avg Profit Margin", f"{avg_margin:.2f}%")

                if "Date" in df.columns:
                    df_monthly = df.resample('ME', on='Date').sum(numeric_only=True)        #Changed M to ME
                    if len(df_monthly) > 1:
                        profit_growth = df_monthly['Profit'].pct_change().mean() * 100
                        rev_growth = df_monthly['Revenue'].pct_change().mean() * 100
                        status = "Growth" if profit_growth > 0 else "Decline"
                        col4.metric("Avg Monthly Profit Change", f"{profit_growth:.2f}%", delta=status)
                    else:
                        st.info("Not enough date data for growth analysis")
            with tab2:
                # Problem Detection
                st.header("⚠️ Problem Detection")
                # Negative Profits
                negative_profits = df[df["Profit"] < 0]
                if not negative_profits.empty:
                    st.warning(f"{len(negative_profits)} items with negative profit. Total loss: ${negative_profits['Profit'].sum():,.2f}")
                    with st.expander("Details"):
                        st.dataframe(negative_profits[["Region", "Product_Category", "Revenue", "Cost", "Profit"]])

                # Worst Region by Profit
                region_profit = df.groupby("Region")["Profit"].sum().sort_values()
                if not region_profit.empty:
                    worst_region = region_profit.index[0]
                    st.warning(f"Worst region: {worst_region} (Profit: ${region_profit.iloc[0]:,.2f})")

                # Worst Product by Margin
                product_margin = df.groupby("Product_Category")["Profit_Margin"].mean().sort_values()
                if not product_margin.empty:
                    worst_product = product_margin.index[0]
                    st.warning(f"Worst product category: {worst_product} (Margin: {product_margin.iloc[0]:.2f}%)")

                # High Revenue but Low Margin
                high_rev_low_margin = df[(df["Revenue"] > df["Revenue"].quantile(0.75)) & (df["Profit_Margin"] < df["Profit_Margin"].quantile(0.25))]
                if not high_rev_low_margin.empty:
                    st.warning(f"{len(high_rev_low_margin)} high-revenue but low-margin items. Avg Margin: {high_rev_low_margin['Profit_Margin'].mean():.2f}%")
                    with st.expander("Details"):
                        st.dataframe(high_rev_low_margin[["Product_Category", "Revenue", "Profit_Margin"]])

                # Cost vs Revenue Growth (if dates)
                if "Date" in df.columns and len(df_monthly) > 1:
                    cost_growth = df_monthly['Cost'].pct_change().mean() * 100
                    rev_growth = df_monthly['Revenue'].pct_change().mean() * 100
                    if cost_growth > rev_growth:
                        st.warning(f"Costs growing faster than revenue ({cost_growth:.2f}% vs {rev_growth:.2f}%). Potential inefficiency.")
            with tab3:
                # Root Cause Analysis
                st.header("🕵️ Root Cause Analysis")
                if not negative_profits.empty:
                    worst_in_neg = negative_profits.groupby("Region")["Profit"].sum().sort_values().index[0]
                    st.info(f"Negative profits mainly in {worst_in_neg} region due to high costs.")
                    if "Supplier" in df.columns:
                        supp_cause = negative_profits.groupby("Supplier")["Cost"].mean().sort_values(ascending=False).index[0]
                        st.info(f"High costs linked to supplier: {supp_cause} (Avg Cost: ${negative_profits[negative_profits['Supplier'] == supp_cause]['Cost'].mean():,.2f})")

                if not region_profit.empty:
                    worst_region_df = df[df["Region"] == worst_region]
                    worst_product_in_region = worst_region_df.groupby("Product_Category")["Profit"].sum().sort_values().index[0]
                    st.info(f"In {worst_region}, worst product: {worst_product_in_region}.")
                    if "Date" in df.columns:
                        monthly_profit = worst_region_df.resample('ME', on='Date')["Profit"].sum()
                        if not monthly_profit.empty:
                            worst_month = monthly_profit.idxmin().strftime("%B %Y") if monthly_profit.idxmin() is not pd.NaT else "N/A"
                            st.info(f"Worst month in {worst_region}: {worst_month} (Profit: ${monthly_profit.min():,.2f}).")

                if not product_margin.empty:
                    worst_product_df = df[df["Product_Category"] == worst_product]
                    worst_region_in_product = worst_product_df.groupby("Region")["Profit_Margin"].mean().sort_values().index[0]
                    st.info(f"For {worst_product}, lowest margin in {worst_region_in_product} region.")

                if not high_rev_low_margin.empty:
                    high_cost_in_low_margin = high_rev_low_margin["Cost"].mean() > df["Cost"].mean()
                    st.info(f"High-rev low-margin items have {'higher' if high_cost_in_low_margin else 'similar'} costs than average. Likely pricing or efficiency issue.")
                    if "Supplier" in df.columns:
                        supp_in_low = high_rev_low_margin.groupby("Supplier")["Cost"].mean().sort_values(ascending=False).index[0]
                        st.info(f"Common supplier in these items: {supp_in_low}.")

            with tab4:
                # Decision Suggestions
                st.header("💡 Actionable Recommendations")
                if total_profit < 0:
                    st.success("Overall losses detected. Prioritize broad cost reduction and review all operations for inefficiencies.")

                if not negative_profits.empty:
                    st.success(f"Profit dropped in {worst_in_neg} region mainly due to high costs. Consider renegotiating with supplier {supp_cause if 'Supplier' in df.columns else ''} or optimizing logistics.")

                if not region_profit.empty:
                    st.success(f"{worst_region} region underperforming. Suggest marketing push or sales training, especially for {worst_product_in_region} in {worst_month if 'Date' in df.columns else 'general periods'}.")

                if "Date" in df.columns and cost_growth > rev_growth:
                    st.success("Costs growing faster than revenue. Audit suppliers and streamline supply chain to control spikes.")

                if not high_rev_low_margin.empty:
                    st.success("High-revenue but low-margin items detected. Review pricing strategy or seek cost efficiencies from suppliers.")

                if not product_margin.empty:
                    st.success(f"For low-margin {worst_product} in {worst_region_in_product}, consider product redesign or supplier change.")
            with tab5:
                # AI Executive Summary
                st.header("🤖 AI Executive Summary")
                load_dotenv()
                if st.button("Generate AI Business Summary"):
                    with st.spinner("Analyzing with AI ..."):
                        
                        # Safe aggregated context (no raw data sent)
                        neg_loss = negative_profits['Profit'].sum() if not negative_profits.empty else 0
                        worst_reg_profit = region_profit.iloc[0] if 'region_profit' in locals() and not region_profit.empty else 0
                        worst_cat_margin = product_margin.iloc[0] if 'product_margin' in locals() and not product_margin.empty else 0
                        
                        context = f"""
                Total Revenue: ${total_revenue:,.0f}
                Total Cost: ${total_cost:,.0f}
                Total Profit: ${total_profit:,.0f} ({'profit' if total_profit >= 0 else 'LOSS'})
                Avg Margin: {avg_margin:.2f}%
                Negative profit items: {len(negative_profits)} (total loss ${neg_loss:,.0f})
                Worst region: {worst_region if 'worst_region' in locals() else 'N/A'} (${worst_reg_profit:,.0f})
                Worst category: {worst_product if 'worst_product' in locals() else 'N/A'} ({worst_cat_margin:.2f}%)
                """

                        if "Date" in df.columns and len(df_monthly) > 1:
                            context += f"\nAvg monthly profit change: {profit_growth:.2f}%"

                        prompt = f"""You are a senior business consultant from Deloitte.
                Analyze this sales performance data and write a concise executive summary (5-8 sentences):

                {context}

                Focus on:
                - Overall health
                - Key problems
                - Root causes
                - 3-5 prioritized actionable recommendations
                Be direct, use numbers, speak in professional business language."""

                        try:
                            # Initialize client with GitHub Models
                            client = OpenAI(
                                base_url="https://models.github.ai/inference",
                                api_key=os.getenv("GITHUB_TOKEN")   # ← your ghp_ token
                            )
                            
                            response = client.chat.completions.create(
                                model="Meta-Llama-3.1-8B-Instruct",   # free & reliable open model
                                # model="openai/gpt-4o-mini"          # try this if available
                                messages=[{"role": "user", "content": prompt}],
                                temperature=0.7,
                                max_tokens=500
                            )
                            
                            summary = response.choices[0].message.content.strip()
                            st.markdown("### AI Summary\n" + summary)
                            
                        except Exception as e:
                            st.error(f"AI generation failed: {str(e)}")
                            st.info("Tip: Make sure GITHUB_TOKEN is set correctly in .env or secrets. "
                                    "Try a different model name from https://github.com/marketplace/models")
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
else:
    st.info("Upload a CSV to start analyzing.")
