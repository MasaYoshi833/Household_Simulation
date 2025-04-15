# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 00:48:46 2025

@author: my199
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="家計＆資産運用シミュレーション", layout="centered")

st.title("家計＆資産運用シミュレーション")

# 初期設定
start_age = st.slider("現在の年齢", min_value=20, max_value=60, value=30)
retirement_age = 65
n_years = retirement_age - start_age
ages = np.arange(start_age, retirement_age + 1)
years = np.arange(2025, 2025 + len(ages))

st.header("🏡 Step 1: 家計シミュレーション")

initial_savings = st.number_input("現在の預金額（万円）", value=300, step=10)
annual_income = st.number_input("現在の年収（万円）", value=500, step=10)
monthly_expense = st.number_input("月々の生活費（万円）", value=20, step=1)

with st.expander("👶 養育費（子供ごとに設定）"):
    num_children = st.selectbox("子供の人数", [0, 1, 2])
    child_birth_ages = []
    for i in range(num_children):
        age = st.slider(f"子供{i+1}の出産時の親の年齢", min_value=start_age, max_value=60, value=start_age+2)
        child_birth_ages.append(age)

loan_amount = st.number_input("住宅ローン借入額（万円）", value=3000, step=100)
loan_interest_rate = st.number_input("ローン金利（年率 %）", value=1.0, step=0.1) / 100
loan_years = st.number_input("返済期間（年）", value=35, step=1)

insurance_monthly = st.number_input("保険料（月額万円）", value=1.0, step=0.1)

pension_start_age = 65
pension_annual = 200
retirement_payout = 2000
income_growth_rate = 0.01
insurance_until_age = 65
child_support_until = 22
child_cost_per_month = 10

def calc_annual_loan_payment(principal, annual_rate, years):
    monthly_rate = annual_rate / 12
    n_payments = years * 12
    monthly_payment = principal * (monthly_rate * (1 + monthly_rate)**n_payments) / ((1 + monthly_rate)**n_payments - 1)
    return monthly_payment * 12

loan_annual_payment = calc_annual_loan_payment(loan_amount, loan_interest_rate, loan_years)

balances = []
incomes = []
expenses = []

balance = initial_savings

for i, age in enumerate(ages):
    year_index = age - start_age

    if age < retirement_age:
        income = annual_income * ((1 + income_growth_rate) ** year_index)
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

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(ages, balances, label='家計のみ（投資なし）', color='gray', linestyle='--')

xtick_indices = [i for i, age in enumerate(ages) if age % 5 == 0 or age == start_age]
xticks = ages[xtick_indices]
xticklabels = [f"{age}\n({year})" for age, year in zip(ages[xtick_indices], years[xtick_indices])]
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels, fontsize=10)

ax.set_title("Household Simulation")
ax.set_xlabel("Age (Year)")
ax.set_ylabel("Amount(10,000Yen)")
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend()
st.pyplot(fig)

# セッションに家計データ保存
st.session_state['household_balance'] = balances
st.session_state['ages'] = ages.tolist()
st.session_state['years'] = years.tolist()

# 次へボタン
if st.button("📈 Step 2: 資産運用シミュレーションへ進む"):
    st.session_state['go_to_investment'] = True
    st.experimental_rerun()

# ----------------------------------------------------
# Step 2: 資産運用シミュレーション（セッションに進む指示がある場合）
# ----------------------------------------------------
if 'go_to_investment' in st.session_state and st.session_state['go_to_investment']:

    st.header("📈 Step 2: 資産運用シミュレーション")

    monthly_contribution = st.slider("月額積立額（万円）", min_value=1, max_value=30, value=5)
    equity_ratio = st.slider("株式比率（残りは債券）（%）", 0, 100, 50)

    equity_return = 0.055
    bond_return = 0.009
    volatilityYearly = np.array([0.23, 0.03])
    correlation = -0.3
    inflation = 0.02

    returnYearly = np.array([equity_return, bond_return])
    corrYearly = np.array([[1, correlation], [correlation, 1]])
    monthly_returns = returnYearly / 12
    monthly_volatility = volatilityYearly / np.sqrt(12)
    cov_matrix = np.diag(monthly_volatility) @ corrYearly @ np.diag(monthly_volatility)
    weights = np.array([equity_ratio / 100, 1 - (equity_ratio / 100)])
    n_simulations = 1000
    n_months = n_years * 12

    all_trajectories = np.zeros((n_simulations, n_years + 1))
    for i in range(n_simulations):
        portfolio_value = 0
        values_by_year = [portfolio_value]
        returns = np.random.multivariate_normal(monthly_returns, cov_matrix, n_months)
        for month in range(n_months):
            monthly_return = np.dot(weights, returns[month])
            portfolio_value *= (1 + monthly_return)
            portfolio_value += monthly_contribution
            if (month + 1) % 12 == 0:
                values_by_year.append(portfolio_value)
        all_trajectories[i, :] = values_by_year

    final_values = all_trajectories[:, -1]
    p25_val = np.percentile(final_values, 25)
    p50_val = np.percentile(final_values, 50)
    p75_val = np.percentile(final_values, 75)

    idx_25 = np.abs(final_values - p25_val).argmin()
    idx_50 = np.abs(final_values - p50_val).argmin()
    idx_75 = np.abs(final_values - p75_val).argmin()

    trajectory_25 = all_trajectories[idx_25]
    trajectory_50 = all_trajectories[idx_50]
    trajectory_75 = all_trajectories[idx_75]

    household_balance = np.array(st.session_state['household_balance'])
    combined_trajectory = household_balance + trajectory_50

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(ages, combined_trajectory, label='家計 + 投資リターン (50%)', color='blue', linewidth=2)
    ax2.plot(ages, household_balance, label='家計のみ（投資なし）', color='gray', linestyle='--')

    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xticklabels, fontsize=10)
    ax2.set_title("Combined Household & Investment Simulation")
    ax2.set_xlabel("Age (Year)")
    ax2.set_ylabel("Amount(10,000Yen)")
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()
    st.pyplot(fig2)

    st.markdown("### 💰 最終積立額（定年時）")
    st.metric("75パーセンタイル", f"{trajectory_75[-1]:,.0f} 万円")
    st.metric("50パーセンタイル（中央値）", f"{trajectory_50[-1]:,.0f} 万円")
    st.metric("25パーセンタイル", f"{trajectory_25[-1]:,.0f} 万円")
    st.metric("貯金のみの場合", f"{monthly_contribution * 12 * n_years:,.0f} 万円")

    st.markdown(f"""
    **📌注釈**：
    - 年金は65歳以降、年間200万円を受給。
    - 退職金は65歳時点で一括2,000万円を受領。
    - 投資リターンの中央値と家計残高を合算表示。
    - 投資額（{monthly_contribution}万円）は家計の支出に含まれています。
    """)
