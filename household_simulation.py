# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 00:48:46 2025

@author: my199
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# 初期設定
st.set_page_config(page_title="資産運用シミュレーション", layout="centered")

st.title("資産運用シミュレーション")

# ----------------------------
# 🧾 前提条件の表示
# ----------------------------
with st.expander("📌 前提条件"):
    st.markdown("""
    - *株式の期待リターン*：5.5%
    - *株式のリスク（年率）*：23%
    - *債券の期待リターン*：0.9%
    - *債券のリスク（年率）*：3%
    - *株式と債券の相関*：-0.3
    - *インフレ率*：2%
    """)

# ----------------------------
# 🎯 入力項目
# ----------------------------
st.subheader("🔧初期設定")

start_age = st.slider("現在の年齢", min_value=20, max_value=60, value=30)
monthly_contribution = st.slider("月額積立額（万円）", min_value=1, max_value=30, value=5)
equity_ratio = st.slider("株式比率(残りは債券)（%）", 0, 100, 50)

# 実行ボタン
if st.button("シミュレーションを実行",type = "primary"):

    # ----------------------------
    # 📊 パラメータ設定
    # ----------------------------
    retirement_age = 65
    start_year = 2025
    end_age = retirement_age
    n_years = end_age - start_age
    n_months = n_years * 12
    ages = np.arange(start_age, end_age + 1)
    years = np.arange(start_year, start_year + n_years + 1)
    
    equity_return = 0.055
    bond_return = 0.009
    inflation = 0.02

    real_equity_return = equity_return
    real_bond_return = bond_return

    returnYearly = np.array([real_equity_return, real_bond_return])
    volatilityYearly = np.array([0.23, 0.03])
    correlation = -0.3
    corrYearly = np.array([[1, correlation],
                           [correlation, 1]])

    # 月次変換
    monthly_returns = returnYearly / 12
    monthly_volatility = volatilityYearly / np.sqrt(12)
    cov_matrix = np.diag(monthly_volatility) @ corrYearly @ np.diag(monthly_volatility)

    # 投資設定
    weights = np.array([equity_ratio / 100, 1 - (equity_ratio / 100)])
    n_simulations = 1000

    all_trajectories = np.zeros((n_simulations, n_years + 1))  # 年単位

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

    # ----------------------------
    # 📉 パーセンタイル計算
    # ----------------------------
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

    # ----------------------------
    # 💹 グラフ描画
    # ----------------------------
    fig, ax = plt.subplots(figsize=(12, 8))

    for i in range(n_simulations):
        ax.plot(ages, all_trajectories[i], color='gray', alpha=0.03)

    ax.plot(ages, trajectory_75, color='blue', linestyle='dashed', linewidth=2, label='75th Percentile')
    ax.plot(ages, trajectory_50, color='red', linewidth=2, label='50th Percentile')
    ax.plot(ages, trajectory_25, color='blue', linestyle='dashed', linewidth=2, label='25th Percentile')

    # 貯金ケース
    saving_trajectory = monthly_contribution * 12 * (ages - start_age)
    ax.plot(ages, saving_trajectory, color='green', linewidth=2, label='Saving Only')

    # 年齢と西暦を両方表示
    xtick_indices = [i for i, age in enumerate(ages) if age % 5 == 0 or age == start_age]
    xticks = ages[xtick_indices]
    xticklabels = [f"{age}\n({year})" for age, year in zip(ages[xtick_indices], years[xtick_indices])]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=10)
    
    # Y軸の上限を85パーセンタイルで設定
    y_max = np.percentile(final_values, 85)
    ax.set_ylim(0, y_max * 1.05)  # 少し余白
    
    ax.set_xlabel("Age(Year)")
    ax.set_ylabel("Amount (10,000 Yen)")
    ax.set_title("Investment Simulation")
    ax.legend()
    st.pyplot(fig)

    # ----------------------------
    # 🧾 結果数値の表示
    # ----------------------------
    st.markdown("### 💰 最終積立額（定年時）")
    st.metric("75パーセンタイル", f"{trajectory_75[-1]:,.0f} 万円")
    st.metric("50パーセンタイル（中央値）", f"{trajectory_50[-1]:,.0f} 万円")
    st.metric("25パーセンタイル", f"{trajectory_25[-1]:,.0f} 万円")
    st.metric("貯金のみの場合", f"{saving_trajectory[-1]:,.0f} 万円")
    
    
# ----------------------------
# ▶ ステップ2：家計管理シミュレーション
# ----------------------------

if st.button("家計管理に進む", type="secondary"):
    st.subheader("🏠 家計シミュレーション")

    # 入力フォーム
    initial_savings = st.number_input("現在の預金額（万円）", value=500)
    annual_income = st.number_input("年収（万円）", value=600)
    monthly_expense = st.number_input("生活費（万円/月）", value=20)
    insurance_monthly = st.number_input("保険（月額・万円）", value=1.5)

    st.markdown("---")
    st.markdown("#### 👶 養育費の設定")
    num_children = st.selectbox("子どもの人数", [0, 1, 2], index=1)
    child_birth_years = []
    for i in range(num_children):
        birth = st.slider(f"子ども{i+1}の出産年齢", min_value=start_age, max_value=60, value=start_age + 2 * i)
        child_birth_years.append(birth)

    st.markdown("---")
    st.markdown("#### 🏠 住宅ローン")
    loan_amount = st.number_input("借入額（万円）", value=3000)
    loan_interest_rate = st.number_input("金利（年率・%）", value=1.0) / 100
    loan_years = st.number_input("返済期間（年）", value=35)

    st.markdown("---")
    st.markdown("#### 👴 年金・退職金")
    pension_annual = 180  # 万円
    retirement_payout = 2000  # 万円
    retirement_age = 65
    pension_start_age = 65
    insurance_until_age = 65
    child_support_until = 22
    child_cost_per_month = 10  # 万円
    income_growth_rate = 0.01

    # 年齢と西暦
    simulation_years = np.arange(start_age, 100)
    simulation_length = len(simulation_years)
    simulation_calendar = np.arange(2025, 2025 + simulation_length)

    # ローン返済額計算
    def calc_annual_loan_payment(principal, annual_rate, years):
        monthly_rate = annual_rate / 12
        n_payments = years * 12
        monthly_payment = principal * (monthly_rate * (1 + monthly_rate)**n_payments) / ((1 + monthly_rate)**n_payments - 1)
        return monthly_payment * 12

    loan_annual_payment = calc_annual_loan_payment(loan_amount, loan_interest_rate, loan_years)

    # 初期化
    balance = initial_savings
    balances = []
    incomes = []
    expenses = []

    for i, age in enumerate(simulation_years):
        year_index = age - start_age

        # ---- 収入 ----
        if age < retirement_age:
            income = annual_income * ((1 + income_growth_rate) ** year_index)
        elif age >= pension_start_age:
            income = pension_annual
        else:
            income = 0

        # ---- 支出 ----
        expense = monthly_expense * 12
        if age <= insurance_until_age:
            expense += insurance_monthly * 12

        child_support = 0
        for birth_year in child_birth_years:
            if birth_year <= age < birth_year + child_support_until:
                child_support += child_cost_per_month * 12
        expense += child_support

        if year_index < loan_years:
            expense += loan_annual_payment

        if age == retirement_age:
            income += retirement_payout

        # 残高更新
        balance = balance + income - expense
        balances.append(balance)
        incomes.append(income)
        expenses.append(expense)

    # グラフ描画
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(simulation_years, balances, label="残高の推移", color="blue")
    ax2.plot(simulation_years, incomes, label="収入", linestyle="--", color="green")
    ax2.plot(simulation_years, expenses, label="支出", linestyle=":", color="red")
    ax2.set_title("家計収支シミュレーション")
    ax2.set_xlabel("年齢 (西暦)")
    xtick_indices = [i for i, age in enumerate(simulation_years) if age % 5 == 0 or age == start_age]
    ax2.set_xticks(simulation_years[xtick_indices])
    ax2.set_xticklabels([f"{age}\n({year})" for age, year in zip(simulation_years[xtick_indices], simulation_calendar[xtick_indices])], fontsize=9)
    ax2.set_ylabel("金額（万円）")
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)

    # 注釈：社会保険料は含まれていませんが、生活費から一定割合が含まれると仮定しています。
    st.caption("※社会保険料は生活費に含まれていると想定。投資額は年60万円で定年まで。年金・退職金は平均的な数値。")
