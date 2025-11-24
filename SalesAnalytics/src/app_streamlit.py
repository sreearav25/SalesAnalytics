"""Streamlit dashboard for sales, salary and profit analytics with main navigation screen.

Author: Aravind
"""

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from company import Company, Employee
from analysis import load_financial_data, basic_kpis, train_sales_model


# --------- Project paths ---------

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# --------- Data loading helpers ---------


@st.cache_data
def build_company(financial_path: str, employees_path: str) -> Company:
    df_fin = load_financial_data(financial_path)
    df_emp = pd.read_csv(employees_path)

    company = Company(name="Demo Company")
    company.load_employees_from_df(df_emp)
    company.load_financials_from_df(df_fin)

    company.init_db()
    company.db_sync_employees_from_memory()
    company.db_sync_financials_from_memory()

    return company


@st.cache_data
def get_financial_df(financial_path: str) -> pd.DataFrame:
    return load_financial_data(financial_path)


# --------- Page sections ---------


def page_home(company: Company, df_fin: pd.DataFrame):
    st.title("Sales Analytics Control Panel")
    st.caption("Author: Aravind")

    st.markdown(
        """
        This is the **main screen**. Use the sidebar to navigate to:

        - 📊 **Company Overview** – high-level KPIs and company summary  
        - 💰 **Salary & Profit Simulator** – adjust salary % with a slider and see impact  
        - 📈 **Charts & Trends** – visualise revenue, profit and salary over time  
        - 🤖 **Model & Predictions** – train the revenue prediction model and review results  
        - 🧑‍💼 **Employee Management** – add / update / delete employees (DB-backed)  
        - 📚 **Resources & Notebooks** – quick links and paths to Jupyter notebooks and data files  
        """
    )

    kpis = basic_kpis(df_fin)
    st.subheader("Quick KPIs")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Revenue", f"{kpis['total_revenue']:,.0f}")
    col2.metric("Total Profit", f"{kpis['total_profit']:,.0f}")
    col3.metric("Avg Profit Margin", f"{kpis['avg_profit_margin']:.2%}")
    col4.metric("Avg Salary Expense", f"{kpis['avg_salary_expense']:,.0f}")

    st.markdown("---")

    st.subheader("Quick Actions")

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("📊 Go to Company Overview"):
            st.session_state["nav_target"] = "Company Overview"
    with c2:
        if st.button("💰 Open Salary Simulator"):
            st.session_state["nav_target"] = "Salary & Profit Simulator"
    with c3:
        if st.button("📈 View Charts & Trends"):
            st.session_state["nav_target"] = "Charts & Trends"

    st.markdown("---")

    st.subheader("How to Open Notebook & CLI from Terminal")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Open main Jupyter notebook:**")
        st.code(
            "jupyter notebook notebooks/01_sales_salary_analytics_demo.ipynb",
            language="bash",
        )
        if st.button("Copy notebook command (visual)"):
            st.info(
                "Use the above command in your terminal to open "
                "`01_sales_salary_analytics_demo.ipynb`."
            )

    with col_right:
        st.markdown("**Run CLI app:**")
        st.code(
            "python src/main_app.py",
            language="bash",
        )
        if st.button("CLI help"):
            st.info(
                "This runs the text-based version of the app in your terminal and "
                "prints KPIs, salary summary, and model predictions."
            )


def page_overview(company: Company, df_fin: pd.DataFrame):
    st.header("📊 Company Overview")
    st.text(company.summary())

    kpis = basic_kpis(df_fin)
    st.subheader("Key KPIs")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Revenue", f"{kpis['total_revenue']:,.0f}")
    col2.metric("Total Profit", f"{kpis['total_profit']:,.0f}")
    col3.metric("Avg Profit Margin", f"{kpis['avg_profit_margin']:.2%}")
    col4.metric("Avg Salary Expense", f"{kpis['avg_salary_expense']:,.0f}")

    st.subheader("Sample Monthly Profit Table")
    st.dataframe(df_fin[["date", "revenue", "profit"]].head())


# ---------- SALARY + TAX + OTHER EXPENSES SIMULATOR ----------

def page_salary_sim(company: Company, df_fin: pd.DataFrame):
    st.header("💰 Salary, Tax & Expense Simulator")

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        salary_pct = st.slider(
            "Change base salaries (%)",
            min_value=-10,
            max_value=30,
            value=5,
            step=1,
            help="Simulate impact of increasing or decreasing employee salaries.",
        )
    with col_s2:
        tax_rate_pct = st.slider(
            "Corporate tax rate (%)",
            min_value=0,
            max_value=50,
            value=20,
            step=1,
            help="Approximate tax rate applied to profit before tax.",
        )
    with col_s3:
        other_exp_pct = st.slider(
            "Change other operating expenses (%)",
            min_value=-30,
            max_value=30,
            value=0,
            step=1,
            help="Simulate changes in general operating expenses (rent, marketing, admin, etc.).",
        )

    salary_factor = 1 + salary_pct / 100.0
    other_factor = 1 + other_exp_pct / 100.0
    tax_rate = tax_rate_pct / 100.0

    df = df_fin.copy()

    df["salary_expense_adj"] = df["salary_expense"] * salary_factor
    df["other_expense_adj"] = df["other_expense"] * other_factor

    df["profit_before_tax_current"] = (
        df["revenue"] - df["cogs"] - df["salary_expense"] - df["other_expense"]
    )
    df["profit_before_tax_adj"] = (
        df["revenue"] - df["cogs"] - df["salary_expense_adj"] - df["other_expense_adj"]
    )

    df["tax_current"] = df["profit_before_tax_current"].clip(lower=0) * 0.20
    df["tax_adj"] = df["profit_before_tax_adj"].clip(lower=0) * tax_rate

    df["profit_after_tax_current"] = df["profit_before_tax_current"] - df["tax_current"]
    df["profit_after_tax_adj"] = df["profit_before_tax_adj"] - df["tax_adj"]

    total_salary_current = df["salary_expense"].sum()
    total_salary_adj = df["salary_expense_adj"].sum()

    total_other_current = df["other_expense"].sum()
    total_other_adj = df["other_expense_adj"].sum()

    total_tax_current = df["tax_current"].sum()
    total_tax_adj = df["tax_adj"].sum()

    total_profit_current = df["profit_after_tax_current"].sum()
    total_profit_adj = df["profit_after_tax_adj"].sum()
    profit_delta = total_profit_adj - total_profit_current

    st.subheader("Expense & Profit Impact (Totals over period)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Total salary expense (current)",
        f"{total_salary_current:,.0f}",
        delta=f"{total_salary_adj - total_salary_current:,.0f}",
    )
    c2.metric(
        "Other operating expenses (current)",
        f"{total_other_current:,.0f}",
        delta=f"{total_other_adj - total_other_current:,.0f}",
    )
    c3.metric(
        "Tax expense (current @20%)",
        f"{total_tax_current:,.0f}",
        delta=f"{total_tax_adj - total_tax_current:,.0f}",
    )
    c4.metric(
        "Profit after tax (current)",
        f"{total_profit_current:,.0f}",
        delta=f"{profit_delta:,.0f}",
    )

    st.markdown("---")

    st.subheader("Summary table (current vs adjusted)")
    summary_df = pd.DataFrame(
        {
            "metric": [
                "Total salary expense",
                "Total other operating expense",
                "Total tax expense",
                "Total profit after tax",
            ],
            "current": [
                total_salary_current,
                total_other_current,
                total_tax_current,
                total_profit_current,
            ],
            "adjusted": [
                total_salary_adj,
                total_other_adj,
                total_tax_adj,
                total_profit_adj,
            ],
        }
    )
    summary_df["difference"] = summary_df["adjusted"] - summary_df["current"]
    st.dataframe(
        summary_df.style.format(
            {"current": ",.0f", "adjusted": ",.0f", "difference": ",.0f"}
        )
    )


def page_charts(company: Company, df_fin: pd.DataFrame):
    st.header("📈 Charts & Trends")

    st.subheader("Revenue vs Profit Over Time")
    chart_df = df_fin[["date", "revenue", "profit"]].set_index("date")
    st.line_chart(chart_df)

    st.subheader("Salary Expense Over Time")
    salary_chart_df = df_fin[["date", "salary_expense"]].set_index("date")
    st.line_chart(salary_chart_df)

    st.subheader("Profit Margin Over Time")
    margin_chart_df = df_fin[["date", "profit_margin"]].set_index("date")
    st.line_chart(margin_chart_df)


def page_model(company: Company, df_fin: pd.DataFrame):
    st.header("🤖 Sales Prediction Model")

    st.write(
        "This trains a simple Linear Regression model to predict **revenue** from calendar "
        "features and expense data."
    )

    if st.button("Train / Retrain Model"):
        model, mae, X_test, y_test, y_pred = train_sales_model(df_fin)
        st.success(f"Model trained. Test MAE: {mae:,.2f}")

        comparison = (
            pd.DataFrame({"actual_revenue": y_test, "predicted_revenue": y_pred})
            .reset_index(drop=True)
        )
        st.subheader("Sample Predictions vs Actual")
        st.dataframe(comparison.head())

    st.info(
        "For deeper model exploration, open the Jupyter notebook "
        "`notebooks/01_sales_salary_analytics_demo.ipynb`."
    )


def page_employee_management(company: Company, df_fin: pd.DataFrame):
    st.header("🧑‍💼 Employee Management (DB-backed)")

    st.subheader("Current Employees (from DB)")
    try:
        df_emp_db = company.db_list_employees()
        st.dataframe(df_emp_db)
    except Exception as e:
        st.error(f"Error reading employees from DB: {e}")
        df_emp_db = pd.DataFrame()

    st.markdown("---")

    st.subheader("➕ Add / Update Employee")

    with st.form("add_update_employee"):
        col1, col2 = st.columns(2)
        with col1:
            employee_id = st.number_input("Employee ID", min_value=1, step=1)
            name = st.text_input("Name")
            department = st.text_input("Department", value="Sales")
        with col2:
            base_salary = st.number_input("Base Salary", min_value=0.0, step=1000.0)
            bonus_rate = st.number_input(
                "Bonus Rate (e.g. 0.10 = 10%)",
                min_value=0.0,
                max_value=1.0,
                step=0.01,
            )
            overtime_hours = st.number_input(
                "Overtime Hours", min_value=0.0, step=1.0
            )
            overtime_rate = st.number_input(
                "Overtime Rate per Hour", min_value=0.0, step=1.0
            )

        submitted = st.form_submit_button("Save Employee")
        if submitted:
            if not name or not department:
                st.error("Name and department are required.")
            else:
                emp = Employee(
                    employee_id=int(employee_id),
                    name=name,
                    department=department,
                    base_salary=float(base_salary),
                    bonus_rate=float(bonus_rate),
                    overtime_hours=float(overtime_hours),
                    overtime_rate=float(overtime_rate),
                )
                try:
                    company.db_insert_or_update_employee(emp)
                    company.db_load_employees_to_memory()
                    st.success(f"Employee {employee_id} saved to database.")
                except Exception as e:
                    st.error(f"Error saving employee: {e}")

    st.markdown("---")

    st.subheader("💷 Update Salary for Existing Employee")

    with st.form("update_salary_form"):
        emp_id_update = st.number_input(
            "Employee ID to update",
            min_value=1,
            step=1,
            key="update_emp_id",
        )
        new_base_salary = st.number_input(
            "New Base Salary",
            min_value=0.0,
            step=1000.0,
            key="update_base_salary",
        )
        update_submitted = st.form_submit_button("Update Salary")

        if update_submitted:
            try:
                existing = company.db_get_employee(int(emp_id_update))
                if existing is None:
                    st.error(f"No employee found with ID {int(emp_id_update)}")
                else:
                    updated = Employee(
                        employee_id=existing.employee_id,
                        name=existing.name,
                        department=existing.department,
                        base_salary=float(new_base_salary),
                        bonus_rate=existing.bonus_rate,
                        overtime_hours=existing.overtime_hours,
                        overtime_rate=existing.overtime_rate,
                    )
                    company.db_insert_or_update_employee(updated)
                    company.db_load_employees_to_memory()
                    st.success(
                        f"Updated base salary for employee {int(emp_id_update)}."
                    )
            except Exception as e:
                st.error(f"Error updating salary: {e}")

    st.markdown("---")

    st.subheader("🗑 Delete Employee")

    with st.form("delete_employee_form"):
        emp_id_delete = st.number_input(
            "Employee ID to delete",
            min_value=1,
            step=1,
            key="delete_emp_id",
        )
        confirm = st.checkbox("I confirm I want to delete this employee.")
        delete_submitted = st.form_submit_button("Delete Employee")

        if delete_submitted:
            if not confirm:
                st.warning("Please confirm deletion by ticking the checkbox.")
            else:
                try:
                    company.db_delete_employee(int(emp_id_delete))
                    company.db_load_employees_to_memory()
                    st.success(
                        f"Employee {int(emp_id_delete)} deleted from database."
                    )
                except Exception as e:
                    st.error(f"Error deleting employee: {e}")


def page_resources(company: Company, df_fin: pd.DataFrame):
    st.header("📚 Resources & Notebooks")

    notebook_path = PROJECT_ROOT / "notebooks" / "01_sales_salary_analytics_demo.ipynb"
    notebook_uri = notebook_path.resolve().as_uri()

    st.markdown("### 📓 Open Jupyter Notebook")
    st.markdown(f"[👉 Click here to open the Jupyter Notebook]({notebook_uri})")

    st.info(
        """
        This will open the exact notebook:
        **notebooks/01_sales_salary_analytics_demo.ipynb**  

        ⚠️ Jupyter Notebook or Jupyter Lab should be running for the browser
        to open it directly. If not, you can start it below.
        """
    )

    if st.button("🚀 Start Jupyter Notebook Server"):
        notebook_dir = PROJECT_ROOT / "notebooks"
        cmd = f'start cmd /k "cd /d {notebook_dir} && jupyter notebook"'
        os.system(cmd)
        st.success("Jupyter Notebook server is starting in a new window...")

    st.markdown("---")

    st.markdown("### 📂 Project Structure (key files)")

    st.markdown(
        """
        - `src/main_app.py` – CLI application entry point  
        - `src/app_streamlit.py` – this Streamlit dashboard  
        - `src/company.py` – core classes (Employee, Company, MonthlyFinancials)  
        - `src/analysis.py` – analytics and prediction model  
        - `data/raw/company_sales_salary.csv` – 3-year monthly sales & salary data  
        - `data/raw/employees.csv` – employee and salary data  
        - `notebooks/01_sales_salary_analytics_demo.ipynb` – main Jupyter notebook  
        """
    )


# --------- Main Streamlit entry point ---------


def main():
    financial_path = PROJECT_ROOT / "data" / "raw" / "company_sales_salary.csv"
    employees_path = PROJECT_ROOT / "data" / "raw" / "employees.csv"

    company = build_company(str(financial_path), str(employees_path))
    df_fin = company.monthly_profit_df()

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        (
            "Home",
            "Company Overview",
            "Salary & Profit Simulator",
            "Charts & Trends",
            "Model & Predictions",
            "Employee Management",
            "Resources & Notebooks",
        ),
    )

    if "nav_target" in st.session_state and st.session_state["nav_target"] != page:
        page = st.session_state["nav_target"]
        st.session_state["nav_target"] = page

    if page == "Home":
        page_home(company, df_fin)
    elif page == "Company Overview":
        page_overview(company, df_fin)
    elif page == "Salary & Profit Simulator":
        page_salary_sim(company, df_fin)
    elif page == "Charts & Trends":
        page_charts(company, df_fin)
    elif page == "Model & Predictions":
        page_model(company, df_fin)
    elif page == "Employee Management":
        page_employee_management(company, df_fin)
    elif page == "Resources & Notebooks":
        page_resources(company, df_fin)


if __name__ == "__main__":
    main()
