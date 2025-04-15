# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 00:09:17 2025

@author: my199
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="家計シミュレーション", layout="centered")
st.title("💡家計シミュレーション")

# 年齢設定
start_age = st.slider("現在の年齢", min_value=20, max_value=60, value=30)
end_age = 100
years = np.arange(start_age, end_age + 1)
n_years = end_age - start_age + 1

# 家計入力（Step1）
st.header("Step 1️⃣ 家計管理の設定")

initial_savings = st.number_input("現在の預金額（万円）", value=300, step=10)
annual_income = st.number_input("現在の年収（万円）", value=600, step=10)
monthly_expense = st.number_input("月々の生活費（万円）", value=25, step=1)

# 子供の入力（改良）
num_children = st.selectbox("子供の人数", [0, 1, 2], index=0)
child_birth_ages = []
if num_children > 0:
    st.markdown("#### 👶 各子供の出生時の親の年齢（現在より前でもOK）")
    for i in range(num_children):
        default_age = start_age if start_age > 25 else 25
        birth_age = st.slider(f"子供{i+1}の出生時の親の年齢", min_value=20, max_value=60, value=default_age)
        child_birth_ages.append(birth_age)

loan_amount = st.number_input("住宅ローン借入額（万円）", value=3000, step=100)
loan_interest_rate = st.number_input("ローン金利（年率 %）", value=1.0, step=0.1) / 100
loan_years = st.number_input("返済期間（年）", value=35, step=1)
insurance_monthly = st.number_input("保険料（月額万円）", value=1.0, step=0.1)

if st.button("✅ 家計シミュレーションを実行"):
    pension_start_age = 65
    pension_annual = 200
    retirement_age = 65
    retirement_payout = 2000
    income_growth_rate = 0.01
    insurance_until_age = 65
    child_support_until = 22
    child_cost_per_month = 10
    contribution_monthly = 5  # 仮に設定（後で変更）

    def calc_annual_loan_payment(principal, annual_rate, years):
        monthly_rate = annual_rate / 12
        n_payments = years * 12
        monthly_payment = principal * (monthly_rate * (1 + monthly_rate)**n_payments) / ((1 + monthly_rate)**n_payments - 1)
        return monthly_payment * 12

    loan_annual_payment = calc_annual_loan_payment(loan_amount, loan_interest_rate, loan_years)

    balance = initial_savings
    balances = []
    incomes = []
    expenses = []

    for i, age in enumerate(years):
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

        balance += income - expense
        balances.append(balance)
        incomes.append(income)
        expenses.append(expense)

    # 注記を先に表示
    st.markdown("""
    ### ℹ️ 前提条件と注記
    - 年収は額面の75%が手取りとして計算されます。
    - 年金は65歳以降、年間200万円。
    - 退職金は65歳で2,000万円を一括受領。
    - 投資シミュレーションの積立は家計支出に含まれます。
    """)

    # 家計グラフ
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(years, balances, label="家計残高", color="gray", linewidth=2)
    ax.plot(years, incomes, label="年収（手取り）", color="green", linestyle='--')
    ax.plot(years, expenses, label="年間支出", color="red", linestyle=':')
    ax.set_title("家計キャッシュフローと残高（100歳まで）")
    ax.set_xlabel("年齢")
    ax.set_ylabel("金額（万円）")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig)

    # セッションに保存
    st.session_state['balances'] = balances
    st.session_state['incomes'] = incomes
    st.session_state['expenses'] = expenses
    st.session_state['years'] = years
    st.session_state['start_age'] = start_age

# Step 2: 資産運用シミュレーション（家計とは独立）
    st.header("Step 2️⃣ 資産運用シミュレーション")

    invest_start_age = st.number_input("投資開始年齢", min_value=start_age, max_value=65, value=start_age)
    invest_end_age = st.number_input("投資終了年齢", min_value=invest_start_age, max_value=65, value=65)
    invest_years = invest_end_age - invest_start_age + 1

    annual_return = st.number_input("想定リターン（年率 %）", value=4.0, step=0.1) / 100
    invest_contribution = st.number_input("年間積立額（万円）", value=60, step=10)

    invest_values = []
    invest_balance = 0
    invest_age_range = np.arange(invest_start_age, invest_end_age + 1)

    for i, age in enumerate(invest_age_range):
        invest_balance *= (1 + annual_return)
        invest_balance += invest_contribution
        invest_values.append(invest_balance)

    # 資産運用だけのグラフ（65歳まで）
    st.subheader("📊 資産運用シミュレーション結果（65歳まで）")

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(invest_age_range, invest_values, color="blue", linewidth=2)
    ax2.set_title("資産運用シミュレーション（65歳まで）")
    ax2.set_xlabel("年齢")
    ax2.set_ylabel("運用残高（万円）")
    ax2.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig2)

    # 統合に向けて保存
    st.session_state['invest_values'] = invest_values
    st.session_state['invest_ages'] = invest_age_range
    st.session_state['invest_start_age'] = invest_start_age

# 家計 + 投資シミュレーション（100歳まで）
st.header("Step 3️⃣ 家計 + 投資シミュレーション（100歳まで）")

if st.button("💡 シミュレーション実行"):
    # 入力取得
    start_age = st.session_state["start_age"]
    current_year = st.session_state["current_year"]
    retirement_age = st.session_state["retirement_age"]
    salary = st.session_state["annual_income"]
    salary_growth = st.session_state["salary_growth_rate"]
    living_expense = st.session_state["living_expense"]
    pension = st.session_state["pension"]
    lump_sum_retirement = st.session_state["lump_sum_retirement"]
    loan_amount = st.session_state["loan_amount"]
    loan_rate = st.session_state["loan_rate"]
    loan_term_years = st.session_state["loan_term_years"]
    loan_start_age = st.session_state["loan_start_age"]
    children_birth_ages = st.session_state.get("children_birth_ages", [])
    monthly_insurance = st.session_state["monthly_insurance"]

    # 投資結果
    invest_ages = st.session_state.get("invest_ages", [])
    invest_values = st.session_state.get("invest_values", [])
    invest_map = dict(zip(invest_ages, invest_values))

    # 年齢のレンジ
    ages = np.arange(start_age, 101)
    n_years = len(ages)

    # ローン返済額
    if loan_amount > 0:
        r = loan_rate
        n = loan_term_years
        annual_loan_payment = loan_amount * r * (1 + r) ** n / ((1 + r) ** n - 1)
    else:
        annual_loan_payment = 0

    # 初期化
    income = np.zeros(n_years)
    expense = np.zeros(n_years)
    balance = np.zeros(n_years)
    cumulative_balance = np.zeros(n_years)
    investment = np.zeros(n_years)
    total_asset = np.zeros(n_years)

    for i, age in enumerate(ages):
        # 収入
        if age < retirement_age:
            income[i] = salary * 0.75  # 手取り
            salary *= (1 + salary_growth)
        elif age == retirement_age:
            income[i] = lump_sum_retirement
        else:
            income[i] = pension

        # 支出
        exp = living_expense + (income[i] * 0.15) + (monthly_insurance * 12)

        if loan_amount > 0 and loan_start_age <= age < loan_start_age + loan_term_years:
            exp += annual_loan_payment

        for birth_age in children_birth_ages:
            child_age = age - birth_age
            if 0 <= child_age < 22:
                exp += 120  # 月10万円×12ヶ月

        expense[i] = exp
        balance[i] = income[i] - expense[i]
        cumulative_balance[i] = cumulative_balance[i - 1] + balance[i] if i > 0 else balance[i]
        investment[i] = invest_map.get(age, invest_values[-1] if invest_values else 0)
        total_asset[i] = cumulative_balance[i] + investment[i]

    # 注記
    st.markdown("**注記：**")
    st.markdown("- 手取りは給与の75%で計算")
    st.markdown(f"- 昇給率：年 {salary_growth * 100:.1f}%")
    st.markdown(f"- 年金：{pension:.0f} 万円／年（65歳から）")
    st.markdown(f"- 退職金：{lump_sum_retirement:.0f} 万円（{retirement_age}歳）")
    st.markdown("- 社会保険料は手取りの15%と仮定")

    # グラフ
    st.subheader("📊 家計 + 投資シミュレーション結果")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(ages, cumulative_balance, label="家計キャッシュフロー", color="green")
    ax.plot(ages, investment, label="運用資産", color="blue", linestyle="--")
    ax.plot(ages, total_asset, label="合計資産", color="orange", linewidth=2)
    ax.set_xlabel("年齢")
    ax.set_ylabel("金額（万円）")
    ax.set_title("年齢別資産推移（100歳まで）")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    st.pyplot(fig)

    # 表
    df = pd.DataFrame({
        "年齢": ages,
        "西暦": current_year + (ages - start_age),
        "収入": income,
        "支出": expense,
        "年間収支": balance,
        "累積収支": cumulative_balance,
        "運用資産": investment,
        "合計資産": total_asset
    })
    st.subheader("📋 年次キャッシュフロー表")
    st.dataframe(df.style.format("{:,.0f}"), use_container_width=True)

