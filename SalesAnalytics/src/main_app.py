"""Full main script with menu + CLI analytics + Streamlit + Jupyter.

Author: Aravind
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from company import Company
from analysis import load_financial_data, basic_kpis, train_sales_model


# --------- PATH SETUP ---------

# This main_app.py is in src/, so:
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

DATA_DIR = PROJECT_ROOT / "data" / "raw"
STREAMLIT_APP = SRC_DIR / "app_streamlit.py"
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "01_sales_salary_analytics_demo.ipynb"


# --------- CORE CLI ANALYTICS (YOUR ORIGINAL LOGIC) ---------

def build_company(financial_path: str, employees_path: str) -> Company:
    """Load data from CSV, build Company object, and sync to SQLite DB."""
    df_fin = load_financial_data(financial_path)
    df_emp = pd.read_csv(employees_path)

    company = Company(name="Demo Company")
    company.load_employees_from_df(df_emp)
    company.load_financials_from_df(df_fin)

    print("\nInitializing SQLite database...")
    company.init_db()

    print("Syncing employees to database...")
    company.db_sync_employees_from_memory()

    print("Syncing financials to database...")
    company.db_sync_financials_from_memory()

    print("Database sync complete.\n")

    return company


def run_cli_analytics():
    """Your original CLI flow, wrapped behind menu option 1."""
    parser = argparse.ArgumentParser(description="Sales & Salary Analytics CLI")
    parser.add_argument(
        "--financial-data",
        type=str,
        default=str(DATA_DIR / "company_sales_salary.csv"),
        help="Path to company sales & salary CSV",
    )
    parser.add_argument(
        "--employees",
        type=str,
        default=str(DATA_DIR / "employees.csv"),
        help="Path to employees CSV",
    )

    # When called from menu, ignore external command-line args
    args = parser.parse_args([])

    company = build_company(args.financial_data, args.employees)

    df_fin = company.monthly_profit_df()
    kpis = basic_kpis(df_fin)

    print(company.summary())
    print("\nKey KPIs:")
    for k, v in kpis.items():
        if "margin" in k:
            print(f"  {k}: {v:.2%}")
        else:
            print(f"  {k}: {v:,.2f}")

    print("\n--- Salary Management ---")
    total_monthly_salary = company.total_monthly_salary_from_employees()
    print(f"Total monthly salary from employees: {total_monthly_salary:,.2f}")

    print("\nDepartment-wise salary breakdown:")
    print(company.department_salary_breakdown())

    delta_5 = company.simulate_salary_increase(0.05)
    print(f"\nImpact of 5% base salary increase on monthly cost: {delta_5:,.2f}")

    print("\nTraining simple revenue prediction model...")
    model, mae, X_test, y_test, y_pred = train_sales_model(df_fin)
    print(f"  Test MAE: {mae:,.2f}")

    comparison = (
        pd.DataFrame({"actual_revenue": y_test, "predicted_revenue": y_pred})
        .reset_index(drop=True)
    )
    print("\nSample prediction vs actual:")
    print(comparison.head())

    print("\nEmployees from DB (first 5):")
    print(company.db_list_employees().head())

    print("\nFinancials from DB (first 5):")
    print(company.db_list_financials().head())

    input("\nPress Enter to return to menu...")


# --------- LAUNCH HELPERS (STREAMLIT, JUPYTER, ETC.) ---------

def clear_full_screen():
    """Make console big and clear (Windows-friendly)."""
    if sys.platform.startswith("win"):
        # Increase console window size (best-effort)
        os.system("mode con: cols=155 lines=40")
        os.system("cls")
    else:
        os.system("clear")


def run_cmd(cmd: str):
    """Run a shell command."""
    print(f"\nRunning: {cmd}\n")
    os.system(cmd)


def run_streamlit_dashboard():
    """Option 2: run Streamlit dashboard."""
    if not STREAMLIT_APP.exists():
        print(f"Streamlit app not found at: {STREAMLIT_APP}")
        input("Press Enter to return to menu...")
        return

    # Run from src directory so imports and paths work
    cmd = f'cd "{SRC_DIR}" && streamlit run "{STREAMLIT_APP.name}"'
    run_cmd(cmd)
    input("\nPress Enter to return to menu...")


def run_jupyter_notebook():
    """Option 3: open Jupyter notebook (specific file if exists)."""
    if NOTEBOOK_PATH.exists():
        cmd = f'jupyter notebook "{NOTEBOOK_PATH}"'
    else:
        cmd = f'jupyter notebook "{PROJECT_ROOT}"'
    run_cmd(cmd)
    input("\nPress Enter to return to menu...")


# --------- MAIN MENU (ALL IN THIS FILE) ---------

def main_menu():
    while True:
        clear_full_screen()
        print("=" * 120)
        print(" " * 35 + "SALES & SALARY ANALYTICS MAIN MENU (Aravind)")
        print("=" * 120)
        print(
            """
    1. Run Python CLI Analytics (this script's logic)
    2. Run Streamlit Dashboard (app_streamlit.py)
    3. Open Jupyter Notebook
    4. Exit
            """
        )
        print("=" * 120)

        choice = input("Select an option (1-4): ").strip()

        if choice == "1":
            clear_full_screen()
            run_cli_analytics()
        elif choice == "2":
            clear_full_screen()
            run_streamlit_dashboard()
        elif choice == "3":
            clear_full_screen()
            run_jupyter_notebook()
        elif choice == "4":
            clear_full_screen()
            print("Exiting... Goodbye!")
            break
        else:
            print("Invalid choice! Press Enter and try again.")
            input()


if __name__ == "__main__":
    main_menu()
