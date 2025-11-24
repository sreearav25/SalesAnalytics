"""Core company and analytics classes with database CRUD.

Author: Aravind
"""

from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path
import sqlite3

import pandas as pd


# ---------- Data classes ----------


@dataclass
class Employee:
    employee_id: int
    name: str
    department: str
    base_salary: float          # monthly base salary for simplicity
    bonus_rate: float = 0.0     # as a fraction, e.g. 0.10 for 10%
    overtime_hours: float = 0.0
    overtime_rate: float = 0.0  # per hour

    @property
    def monthly_bonus(self) -> float:
        return self.base_salary * self.bonus_rate

    @property
    def monthly_overtime_pay(self) -> float:
        return self.overtime_hours * self.overtime_rate

    @property
    def total_monthly_compensation(self) -> float:
        return self.base_salary + self.monthly_bonus + self.monthly_overtime_pay


@dataclass
class MonthlyFinancials:
    date: pd.Timestamp
    revenue: float
    cogs: float
    salary_expense: float
    other_expense: float

    @property
    def profit(self) -> float:
        return self.revenue - self.cogs - self.salary_expense - self.other_expense

    @property
    def profit_margin(self) -> float:
        return self.profit / self.revenue if self.revenue else 0.0


# ---------- Company model with DB CRUD ----------


class Company:
    """Represents the company, holding employees and financial data.

    In-memory:
        - self.employees: List[Employee]
        - self.financials: List[MonthlyFinancials]

    Database (SQLite, optional):
        - employees table
        - financials table
    """

    def __init__(self, name: str, db_path: Optional[str | Path] = None):
        self.name = name
        self.employees: List[Employee] = []
        self.financials: List[MonthlyFinancials] = []

        # SQLite DB file (default: data/company.db)
        if db_path is None:
            self.db_path = Path("data") / "company.db"
        else:
            self.db_path = Path(db_path)

    # ---------- Internal DB helpers ----------

    def _get_connection(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        """Create tables if they do not exist."""
        conn = self._get_connection()
        cur = conn.cursor()

        # Employees table
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS employees (
                employee_id    INTEGER PRIMARY KEY,
                name           TEXT NOT NULL,
                department     TEXT NOT NULL,
                base_salary    REAL NOT NULL,
                bonus_rate     REAL NOT NULL DEFAULT 0.0,
                overtime_hours REAL NOT NULL DEFAULT 0.0,
                overtime_rate  REAL NOT NULL DEFAULT 0.0
            )
            """
        )

        # Financials table
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS financials (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                date           TEXT NOT NULL UNIQUE,
                revenue        REAL NOT NULL,
                cogs           REAL NOT NULL,
                salary_expense REAL NOT NULL,
                other_expense  REAL NOT NULL
            )
            """
        )

        conn.commit()
        conn.close()

    # ---------- Employees & salary management (in-memory) ----------

    def add_employee(self, employee: Employee) -> None:
        self.employees.append(employee)

    def load_employees_from_df(self, df: pd.DataFrame) -> None:
        for _, row in df.iterrows():
            self.add_employee(
                Employee(
                    employee_id=int(row["employee_id"]),
                    name=str(row["name"]),
                    department=str(row["department"]),
                    base_salary=float(row["base_salary"]),
                    bonus_rate=float(row.get("bonus_rate", 0.0)),
                    overtime_hours=float(row.get("overtime_hours", 0.0)),
                    overtime_rate=float(row.get("overtime_rate", 0.0)),
                )
            )

    def total_monthly_salary_from_employees(self) -> float:
        """Sum of all employees' total monthly compensation."""
        return sum(e.total_monthly_compensation for e in self.employees)

    def department_salary_breakdown(self) -> pd.DataFrame:
        """Return a DataFrame of total monthly salary by department."""
        records = {}
        for e in self.employees:
            records.setdefault(e.department, 0.0)
            records[e.department] += e.total_monthly_compensation

        return (
            pd.DataFrame(
                [
                    {"department": dept, "total_monthly_salary": total}
                    for dept, total in records.items()
                ]
            )
            .sort_values("total_monthly_salary", ascending=False)
            .reset_index(drop=True)
        )

    def simulate_salary_increase(self, percent_increase: float) -> float:
        """
        Simulate impact on total monthly salary if base salaries increase
        by a given percent (e.g. 0.05 for +5%). Returns the delta.
        """
        current_total = self.total_monthly_salary_from_employees()
        new_total = sum(
            (e.base_salary * (1 + percent_increase))
            + e.monthly_bonus
            + e.monthly_overtime_pay
            for e in self.employees
        )
        return new_total - current_total

    # ---------- Financials & profit analytics (in-memory) ----------

    def load_financials_from_df(self, df: pd.DataFrame) -> None:
        for _, row in df.iterrows():
            self.financials.append(
                MonthlyFinancials(
                    date=pd.to_datetime(row["date"]),
                    revenue=float(row["revenue"]),
                    cogs=float(row["cogs"]),
                    salary_expense=float(row["salary_expense"]),
                    other_expense=float(row["other_expense"]),
                )
            )

    def monthly_profit_df(self) -> pd.DataFrame:
        """Return a DataFrame of monthly profit and profit margin."""
        records = []
        for f in self.financials:
            records.append(
                {
                    "date": f.date,
                    "revenue": f.revenue,
                    "cogs": f.cogs,
                    "salary_expense": f.salary_expense,
                    "other_expense": f.other_expense,
                    "profit": f.profit,
                    "profit_margin": f.profit_margin,
                }
            )
        return (
            pd.DataFrame(records)
            .sort_values("date")
            .reset_index(drop=True)
        )

    def summary(self) -> str:
        df = self.monthly_profit_df()
        total_revenue = df["revenue"].sum()
        total_profit = df["profit"].sum()
        avg_margin = (total_profit / total_revenue) if total_revenue else 0.0
        return (
            f"Company: {self.name}\n"
            f"Employees: {len(self.employees)}\n"
            f"Total Revenue (all periods): {total_revenue:,.2f}\n"
            f"Total Profit (all periods): {total_profit:,.2f}\n"
            f"Average Profit Margin: {avg_margin:.2%}"
        )

    # ===============================================================
    #                DATABASE CRUD OPERATIONS
    # ===============================================================

    # ---------- EMPLOYEE CRUD (DB) ----------

    def db_insert_or_update_employee(self, employee: Employee) -> None:
        """Insert or update an employee in the database."""
        conn = self._get_connection()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO employees (
                employee_id, name, department, base_salary,
                bonus_rate, overtime_hours, overtime_rate
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(employee_id) DO UPDATE SET
                name = excluded.name,
                department = excluded.department,
                base_salary = excluded.base_salary,
                bonus_rate = excluded.bonus_rate,
                overtime_hours = excluded.overtime_hours,
                overtime_rate = excluded.overtime_rate
            """,
            (
                employee.employee_id,
                employee.name,
                employee.department,
                employee.base_salary,
                employee.bonus_rate,
                employee.overtime_hours,
                employee.overtime_rate,
            ),
        )
        conn.commit()
        conn.close()

    def db_delete_employee(self, employee_id: int) -> None:
        """Delete an employee from the database."""
        conn = self._get_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM employees WHERE employee_id = ?", (employee_id,))
        conn.commit()
        conn.close()

    def db_get_employee(self, employee_id: int) -> Optional[Employee]:
        """Fetch a single employee from the database."""
        conn = self._get_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM employees WHERE employee_id = ?", (employee_id,))
        row = cur.fetchone()
        conn.close()

        if row is None:
            return None

        return Employee(
            employee_id=row["employee_id"],
            name=row["name"],
            department=row["department"],
            base_salary=row["base_salary"],
            bonus_rate=row["bonus_rate"],
            overtime_hours=row["overtime_hours"],
            overtime_rate=row["overtime_rate"],
        )

    def db_list_employees(self) -> pd.DataFrame:
        """Return all employees as a DataFrame from the database."""
        conn = self._get_connection()
        df = pd.read_sql_query("SELECT * FROM employees ORDER BY employee_id", conn)
        conn.close()
        return df

    def db_sync_employees_from_memory(self) -> None:
        """Write current in-memory employees to DB (upsert each one)."""
        for e in self.employees:
            self.db_insert_or_update_employee(e)

    def db_load_employees_to_memory(self) -> None:
        """Load employees from DB into self.employees (overwriting current list)."""
        conn = self._get_connection()
        df = pd.read_sql_query("SELECT * FROM employees ORDER BY employee_id", conn)
        conn.close()

        self.employees = []
        self.load_employees_from_df(df)

    # ---------- FINANCIALS CRUD (DB) ----------

    def db_insert_or_update_financials(self, mf: MonthlyFinancials) -> None:
        """Insert or update a MonthlyFinancials row by date."""
        conn = self._get_connection()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO financials (
                date, revenue, cogs, salary_expense, other_expense
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(date) DO UPDATE SET
                revenue = excluded.revenue,
                cogs = excluded.cogs,
                salary_expense = excluded.salary_expense,
                other_expense = excluded.other_expense
            """,
            (
                mf.date.strftime("%Y-%m-%d"),
                mf.revenue,
                mf.cogs,
                mf.salary_expense,
                mf.other_expense,
            ),
        )
        conn.commit()
        conn.close()

    def db_delete_financials_by_date(self, date: pd.Timestamp) -> None:
        conn = self._get_connection()
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM financials WHERE date = ?",
            (date.strftime("%Y-%m-%d"),),
        )
        conn.commit()
        conn.close()

    def db_list_financials(self) -> pd.DataFrame:
        """Return all financial rows as DataFrame from DB."""
        conn = self._get_connection()
        df = pd.read_sql_query("SELECT * FROM financials ORDER BY date", conn)
        conn.close()
        df["date"] = pd.to_datetime(df["date"])
        return df

    def db_sync_financials_from_memory(self) -> None:
        """Write current in-memory financials to DB (upsert each one)."""
        for mf in self.financials:
            self.db_insert_or_update_financials(mf)

    def db_load_financials_to_memory(self) -> None:
        """Load financials from DB into self.financials (overwriting current list)."""
        df = self.db_list_financials()
        self.financials = []
        self.load_financials_from_df(df)
