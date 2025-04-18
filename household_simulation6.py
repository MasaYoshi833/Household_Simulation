# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 21:22:00 2025

@author: my199
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="家計シミュレーション", layout="centered")
st.title("家計シミュレーション")

# 家計入力（Step1）
st.subheader("🔧初期設定")

start_age = st.slider("現在の年齢", min_value=20, max_value=60, value=30)
retirement_age = 65
start_year = 2025
end_age = 100
n_years = end_age - start_age
n_months = n_years * 12
ages = np.arange(start_age, end_age + 1)
years = np.arange(start_year, start_year + n_years + 1)

initial_savings = st.number_input("現在の預金額（万円）", value=400, step=10)
annual_income = st.number_input("現在の年収（万円）", value=450, step=10)
monthly_expense = st.number_input("月々の生活費（万円）", value=15, step=1)

# 教育費
num_children = st.selectbox("子供の人数", [0, 1, 2, 3], index=0)
child_birth_ages = []
if num_children > 0:
    st.markdown("##子供の出生時の年齢")
    for i in range(num_children):
        default_age = start_age if start_age > 25 else 25
        birth_age = st.slider(f"子供{i+1}の出生時の親の年齢", min_value=20, max_value=60, value=default_age)
        child_birth_ages.append(birth_age)

# 住宅ローン
use_loan = st.checkbox("住宅ローンあり")
if use_loan:
    loan_amount = st.number_input("住宅ローン借入額（万円）", value=3000, step=100)
    loan_interest_rate = st.number_input("ローン金利（年率 %）", value=1.0, step=0.1) / 100
    loan_years = st.number_input("返済期間（年）", value=35, step=1)
else:
    loan_amount = 0
    loan_interest_rate = 0.0
    loan_years = 0

def calc_annual_loan_payment(principal, annual_rate, years):
    monthly_rate = annual_rate / 12
    n_payments = years * 12
    if monthly_rate == 0:
        return principal / years
    monthly_payment = principal * (monthly_rate * (1 + monthly_rate) ** n_payments) / ((1 + monthly_rate) ** n_payments - 1)
    return monthly_payment * 12

loan_annual_payment = calc_annual_loan_payment(loan_amount, loan_interest_rate, loan_years) if use_loan else 0.0

# 保険
use_insurance = st.checkbox("保険加入あり")
insurance_monthly = st.number_input("保険料（月額万円）", value=1.0, step=0.1) if use_insurance else 0.0

if st.button("シミュレーションを実行", type="primary"):
    st.session_state["household_done"] = True

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
            income = gross_income * 0.75
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

    st.session_state["balances"] = balances
    st.session_state["incomes"] = incomes
    st.session_state["expenses"] = expenses
    st.session_state["ages"] = ages
    st.session_state["years"] = years

if st.session_state.get("household_done"):
    st.markdown("""
    📌 注記  
     - 年収は額面の75%を手取り（25%は社会保険料・税金）、昇給率年間１％として計算。  
     - 年金（国民年金のみ）は65歳以降、月5万6千円を受給。  
     - 退職金は65歳で2,000万円を一括受領。  
     - 教育費は子供が22歳になるまで一人当たり月10万円の計算。
    """)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(st.session_state["years"], st.session_state["balances"], label="Balance", color="blue", linewidth=2)
    ax.plot(st.session_state["years"], st.session_state["incomes"], label="Income", color="green", linestyle='--')
    ax.plot(st.session_state["years"], st.session_state["expenses"], label="Expense", color="red", linestyle=':')
    xtick_indices = [i for i, a in enumerate(st.session_state["ages"]) if a % 5 == 0 or a == start_age]
    ax.set_xticks(st.session_state["years"][xtick_indices])
    ax.set_xticklabels([f"{a}\n({y})" for a, y in zip(st.session_state["ages"][xtick_indices], st.session_state["years"][xtick_indices])], rotation=45)
    ax.set_xlabel("Age(Year)")
    ax.set_ylabel("Amount（10,000Yen）")
    ax.set_title("Household Balance & Cashflow")
    ax.legend()
    st.pyplot(fig)

# 資産運用シミュレーション
if st.session_state.get("household_done"):
    st.header("資産運用シミュレーション")

    monthly_contribution = st.slider("月額積立額（万円）", 1, 30, 5)
    equity_ratio = st.slider("株式比率(残りは債券)（%）", 0, 100, 50)

    st.markdown("""
    📌 注記  
      - *株式はリターン：5.5%、リスク:23%（年率）
      - *債券はリターン：0.9%、リスク:3%（年率）
      - *株式と債券の相関*：-0.3
    """)

    if st.button("資産運用シミュレーションを実行", key="run_investment"):
        invest_years = retirement_age - start_age
        invest_months = invest_years * 12
        invest_ages = np.arange(start_age, retirement_age + 1)

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
        all_trajectories = np.zeros((n_simulations, invest_years + 1))

        for i in range(n_simulations):
            portfolio_value = 0
            values_by_year = [portfolio_value]
            returns = np.random.multivariate_normal(monthly_returns, cov_matrix, invest_months)
            for month in range(invest_months):
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
        saving_trajectory = monthly_contribution * 12 * (invest_ages - start_age)

        # 投資グラフ
        fig, ax = plt.subplots(figsize=(12, 8))
        for i in range(n_simulations):
            ax.plot(invest_ages, all_trajectories[i], color='gray', alpha=0.03)
        ax.plot(invest_ages, trajectory_75, color='blue', linestyle='dashed', linewidth=2, label='75th Percentile')
        ax.plot(invest_ages, trajectory_50, color='red', linewidth=2, label='50th Percentile')
        ax.plot(invest_ages, trajectory_25, color='blue', linestyle='dashed', linewidth=2, label='25th Percentile')
        ax.plot(invest_ages, saving_trajectory, color='green', linewidth=2, label='Saving Only')

        xtick_indices = [i for i, a in enumerate(invest_ages) if a % 5 == 0 or a == start_age]
        ax.set_xticks(invest_ages[xtick_indices])
        ax.set_xticklabels([f"{a}\n({start_year + a - start_age})" for a in invest_ages[xtick_indices]])
        ax.set_ylim(0, np.percentile(final_values, 85) * 1.05)
        ax.set_xlabel("Age(Year)")
        ax.set_ylabel("Amount (10,000 Yen)")
        ax.set_title("Investment Simulation")
        ax.legend()
        st.pyplot(fig)

        st.markdown("#　定年時積立額")
        st.metric("75パーセンタイル", f"{trajectory_75[-1]:,.0f} 万円")
        st.metric("50パーセンタイル（中央値）", f"{trajectory_50[-1]:,.0f} 万円")
        st.metric("25パーセンタイル", f"{trajectory_25[-1]:,.0f} 万円")
        st.metric("貯金のみの場合", f"{saving_trajectory[-1]:,.0f} 万円")

        # ✅ 統合グラフ（100歳まで）
        full_trajectory = np.zeros(len(ages))
        full_trajectory[:len(trajectory_50)] = trajectory_50
        household_balances = np.array(st.session_state["balances"])
        integrated = household_balances + full_trajectory

        st.header("統合グラフ（家計 + 資産運用）")
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(ages, integrated, label="Balance (with Investment(Median))", color="blue", linewidth=2)
        ax.plot(ages, household_balances, label="No investment", color="blue", linestyle="--")

        xtick_indices = [i for i, a in enumerate(st.session_state["ages"]) if a % 5 == 0 or a == start_age]
        ax.set_xticks(ages[xtick_indices])
        ax.set_xticklabels([f"{a}\n({start_year + a - start_age})" for a in xtick_indices])
        ax.set_xlabel("Age(Year)")
        ax.set_ylabel("Amount (10,000 Yen)")
        ax.set_title("Integrated Simulation (Household + Investment)")
        ax.legend()
        st.pyplot(fig)

        st.markdown(f"💡 **100歳時点の統合資産額（中央値）**: `{integrated[-1]:,.0f} 万円`")
