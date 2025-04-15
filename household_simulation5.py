# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 00:09:17 2025

@author: my199
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


st.set_page_config(page_title="家計シミュレーション", layout="centered")
st.title("家計シミュレーション")


# 家計入力（Step1）
st.subheader("🔧初期設定")

# 年齢設定
start_age = st.slider("現在の年齢", min_value=20, max_value=60, value=30)
retirement_age = 65
start_year = 2025
end_age = 100
n_years = end_age - start_age
n_months = n_years * 12
ages = np.arange(start_age, end_age + 1)
years = np.arange(start_year, start_year + n_years + 1)

# 貯蓄・給与
initial_savings = st.number_input("現在の預金額（万円）", value=400, step=10)
annual_income = st.number_input("現在の年収（万円）", value=450, step=10)
monthly_expense = st.number_input("月々の生活費（万円）", value=15, step=1)

# 養育費
num_children = st.selectbox("子供の人数", [0, 1, 2], index=0)
child_birth_ages = []
if num_children > 0:
    st.markdown("#### 各子供の出生時の年齢")
    for i in range(num_children):
        default_age = start_age if start_age > 25 else 25
        birth_age = st.slider(f"子供{i+1}の出生時の親の年齢", min_value=20, max_value=60, value=default_age)
        child_birth_ages.append(birth_age)

#　住宅ローン
use_loan = st.checkbox("住宅ローンあり")
if use_loan:
    loan_amount = st.number_input("住宅ローン借入額（万円）", value=3000, step=100)
    loan_interest_rate = st.number_input("ローン金利（年率 %）", value=1.0, step=0.1) / 100
    loan_years = st.number_input("返済期間（年）", value=35, step=1)
else:
    loan_amount = 0
    loan_interest_rate = 0.0
    loan_years = 0

# 住宅ローン年間返済額（ローンがある場合のみ計算）
if use_loan and loan_amount > 0 and loan_interest_rate > 0 and loan_years > 0:
    def calc_annual_loan_payment(principal, annual_rate, years):
        monthly_rate = annual_rate / 12
        n_payments = years * 12
        if monthly_rate == 0:
            return principal / years  # 無金利
        monthly_payment = principal * (monthly_rate * (1 + monthly_rate) ** n_payments) / ((1 + monthly_rate) ** n_payments - 1)
        return monthly_payment * 12

    loan_annual_payment = calc_annual_loan_payment(loan_amount, loan_interest_rate, loan_years)
else:
    loan_annual_payment = 0.0


# 保険
use_insurance = st.checkbox("保険加入あり")
if use_insurance:
    insurance_monthly = st.number_input("保険料（月額万円）", value=1.0, step=0.1)
else:
    insurance_monthly = 0.0


if st.button("シミュレーションを実行",type = "primary"):
    pension_start_age = 65
    pension_annual = 67.2
    retirement_payout = 2000
    income_growth_rate = 0.01
    insurance_until_age = 65
    child_support_until = 22
    child_cost_per_month = 10


    balance = initial_savings
    balances = []
    incomes = []
    expenses = []

    for i, age in enumerate(ages):
        year_index = age - start_age

        if age < retirement_age:
            gross_income = annual_income * ((1 + income_growth_rate) ** year_index)
            income = gross_income * 0.75  # 手取り
        elif age >= pension_start_age:
            income = pension_annual
        else:
            income = 0

        expense = monthly_expense * 12
        if age <= insurance_until_age:
            expense += insurance_monthly * 12

        child_support = 0
        for birth_age in child_birth_ages:
            if birth_age <= age < birth_age + child_support_until:
                child_support += child_cost_per_month * 12
        expense += child_support

        if year_index < loan_years:
            expense += loan_annual_payment

        if age == retirement_age:
            income += retirement_payout

        balance = balance + income - expense
        balances.append(balance)
        incomes.append(income)
        expenses.append(expense)

    # 注記を先に表示
    st.markdown("""
    📌 注
     - 年収は昇給率年間１％、額面の75%が手取りとして計算されます。
     - 年金は65歳以降、月5万6千円を受給。
     - 退職金は65歳で2,000万円を一括受領。
     - 養育費は子供が22歳になるまで一人当たり月10万円の計算。
    """)

    # 家計グラフ
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(years, balances, label="Balance", color="blue", linewidth=2)
    ax.plot(years, incomes, label="Income", color="green", linestyle='--')
    ax.plot(years, expenses, label="Expense", color="red", linestyle=':')
    
    # 年齢と西暦を両方表示
    xtick_indices = [i for i, a in enumerate(ages) if a % 5 == 0 or a == start_age]
    xticks = [years[i] for i in xtick_indices]
    xticklabels = [f"{ages[i]}\n({years[i]})" for i in xtick_indices]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=45, ha='right', fontsize=10)
    
    ax.set_title("Household Balance & Cashflow")
    ax.set_xlabel("Age(Year)")
    ax.set_ylabel("Amount（10,000Yen）")
    ax.legend()
    st.pyplot(fig)

    # セッションに保存
    st.session_state['balances'] = balances
    st.session_state['incomes'] = incomes
    st.session_state['expenses'] = expenses
    st.session_state['years'] = years
    st.session_state['start_age'] = start_age

# Step 2: 資産運用シミュレーション（家計とは独立）
 # ---- 資産運用シミュレーション ----
st.header("資産運用シミュレーション")

monthly_contribution = st.slider("月額積立額（万円）", 1, 30, 5)
equity_ratio = st.slider("株式比率(残りは債券)（%）", 0, 100, 50)

if st.button("資産運用シミュレーションを実行", key="run_investment"):
    retirement_age = 65
    end_age = retirement_age
    n_years = end_age - start_age
    n_months = n_years * 12
    ages = np.arange(start_age, end_age + 1)

    equity_return = 0.055
    bond_return = 0.009
    volatilityYearly = np.array([0.23, 0.03])
    correlation = -0.3
    corrYearly = np.array([[1, correlation], [correlation, 1]])

    monthly_returns = np.array([equity_return, bond_return]) / 12
    monthly_volatility = volatilityYearly / np.sqrt(12)
    cov_matrix = np.diag(monthly_volatility) @ corrYearly @ np.diag(monthly_volatility)
    weights = np.array([equity_ratio / 100, 1 - (equity_ratio / 100)])
    n_simulations = 1000
    all_trajectories = np.zeros((n_simulations, n_years + 1))

    for i in range(n_simulations):
        portfolio_value = 0
        values_by_year = [portfolio_value]
        returns = np.random.multivariate_normal(monthly_returns, cov_matrix, n_months)
        for month in range(n_months):
            portfolio_value *= (1 + np.dot(weights, returns[month]))
            portfolio_value += monthly_contribution
            if (month + 1) % 12 == 0:
                values_by_year.append(portfolio_value)
        all_trajectories[i, :] = values_by_year

    final_values = all_trajectories[:, -1]
    p25_val, p50_val, p75_val = np.percentile(final_values, [25, 50, 75])
    trajectory_25 = all_trajectories[np.abs(final_values - p25_val).argmin()]
    trajectory_50 = all_trajectories[np.abs(final_values - p50_val).argmin()]
    trajectory_75 = all_trajectories[np.abs(final_values - p75_val).argmin()]

    fig, ax = plt.subplots(figsize=(12, 8))
    for i in range(n_simulations):
        ax.plot(ages, all_trajectories[i], color='gray', alpha=0.03)
    ax.plot(ages, trajectory_75, color='blue', linestyle='dashed', linewidth=2, label='75th Percentile')
    ax.plot(ages, trajectory_50, color='red', linewidth=2, label='50th Percentile')
    ax.plot(ages, trajectory_25, color='blue', linestyle='dashed', linewidth=2, label='25th Percentile')

    saving_trajectory = monthly_contribution * 12 * (ages - start_age)
    ax.plot(ages, saving_trajectory, color='green', linewidth=2, label='Saving Only')

    xtick_indices = [i for i, a in enumerate(ages) if a % 5 == 0 or a == start_age]
    ax.set_xticks(ages[xtick_indices])
    ax.set_xticklabels([f"{a}\n({start_year + a - start_age})" for a in ages[xtick_indices]])
    ax.set_ylim(0, np.percentile(final_values, 85) * 1.05)
    ax.set_xlabel("Age(Year)")
    ax.set_ylabel("Amount (10,000 Yen)")
    ax.set_title("Investment Simulation")
    ax.legend()
    st.pyplot(fig)

    st.markdown("### 💰 最終積立額（定年時）")
    st.metric("75パーセンタイル", f"{trajectory_75[-1]:,.0f} 万円")
    st.metric("50パーセンタイル（中央値）", f"{trajectory_50[-1]:,.0f} 万円")
    st.metric("25パーセンタイル", f"{trajectory_25[-1]:,.0f} 万円")
    st.metric("貯金のみの場合", f"{saving_trajectory[-1]:,.0f} 万円")

