# Assistant Strategy Builder: Modular AI-Orchestrated Strategy Evaluation

# -----------------------------
# STEP 1: USER DEFINED CONFIG
# -----------------------------
REQUIRED_METRICS = {
    "sharpe": 1.6,
    "max_drawdown": 10
}

# -----------------------------
# STEP 2: IMPORTS
# -----------------------------
import pandas as pd
import numpy as np
from datetime import datetime
from openai import OpenAI

# -----------------------------
# STEP 3: CORE MODULES
# -----------------------------

def run_backtest(strategy_code: str, data_path: str) -> pd.DataFrame:
    """
    Execute the strategy code (as Python logic) on your data.
    Returns a DataFrame with ['timestamp', 'pnl', 'equity_curve']
    """
    local_vars = {}
    exec(strategy_code, globals(), local_vars)
    backtest_func = local_vars.get("backtest_strategy")
    return backtest_func(data_path)

def calculate_metrics(pnl_df: pd.DataFrame) -> dict:
    returns = pnl_df['equity_curve'].pct_change().dropna()
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if not returns.empty else 0
    drawdown = (pnl_df['equity_curve'].cummax() - pnl_df['equity_curve']).max() / pnl_df['equity_curve'].cummax().max() * 100 if not pnl_df.empty else 0
    profit_factor = pnl_df[pnl_df['pnl'] > 0]['pnl'].sum() / abs(pnl_df[pnl_df['pnl'] < 0]['pnl'].sum()) if not pnl_df[pnl_df['pnl'] < 0].empty else np.inf
    return {
        "sharpe": round(sharpe, 2),
        "max_drawdown": round(drawdown, 2),
        "profit_factor": round(profit_factor, 2),
        "final_profit": round(pnl_df['pnl'].sum(), 2) if not pnl_df.empty else 0
    }

def check_thresholds(metrics: dict, required: dict) -> bool:
    return metrics['sharpe'] >= required['sharpe'] and metrics['max_drawdown'] <= required['max_drawdown']

# -----------------------------
# STEP 4: LOOP FOR ASSISTANT
# -----------------------------
def iterate_strategy_with_assistant(data_path: str, assistant_strategy_generator) -> tuple:
    history = []
    while True:
        strategy_code = assistant_strategy_generator(history)
        pnl_df = run_backtest(strategy_code, data_path)
        metrics = calculate_metrics(pnl_df)
        history.append({"code": strategy_code, "metrics": metrics})

        if check_thresholds(metrics, REQUIRED_METRICS):
            return strategy_code, metrics, pnl_df
        else:
            print("Retrying...", metrics)

# -----------------------------
# STEP 5: GPT-BASED STRATEGY GENERATOR (USING OpenAI API)
# -----------------------------
client = OpenAI()
client = OpenAI(api_key=xxxxx)

def assistant_strategy_generator(history):
    framework = """
Use the following backtest function template.
You may modify the logic inside as needed to improve performance, reduce drawdown, or enhance Sharpe ratio.

-------------------------------
def backtest_strategy(data_path):
    import pandas as pd
    import numpy as np
    import pandas_ta as ta
    from datetime import timedelta

    # Load data
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # ------------------------------
    # Add technical indicators here
    # ------------------------------
    # Example: df['rsi'] = ta.rsi(df['close'], length=14)

    # ------------------------------
    # Define signal and entry/exit logic
    # ------------------------------
    # Example: df['signal'] = ...
    # Use your logic to set `entry_price`, `exit_price`, and timestamps

    # ------------------------------
    # Simulate trades and calculate PnL
    # ------------------------------
    trades = []
    position = None
    entry_price = None
    entry_time = None

    for i in range(len(df)):
        # Signal handling logic
        pass

    trades_df = pd.DataFrame(trades)

    # ------------------------------
    # Calculate equity curve
    # ------------------------------
    if not trades_df.empty:
        trades_df['pnl'] = trades_df['Exit_Price'] - trades_df['Entry_Price']
        trades_df['equity_curve'] = trades_df['pnl'].cumsum()
        trades_df['timestamp'] = trades_df['Exit_Time']
    else:
        # Return empty structure if no trades
        trades_df = pd.DataFrame(columns=['timestamp', 'pnl', 'equity_curve'])

    return trades_df[['timestamp', 'pnl', 'equity_curve']]
-------------------------------
"""

    prompt = f"""
You are a trading assistant. Generate a Python backtest function named `backtest_strategy(data_path)`.
It should use the framework provided below to create a breakout trading strategy.
Try to meet a Sharpe ratio > 1.6 and drawdown < 10%.

Framework:
{framework}
"""

    if history:
        last_metrics = history[-1]["metrics"]
        prompt += f"\nLast attempt metrics: Sharpe = {last_metrics['sharpe']}, Drawdown = {last_metrics['max_drawdown']}%"
        prompt += "\nRevise the strategy logic to improve the metrics."

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a quant trading assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# -----------------------------
# STEP 6: RUN IT
# -----------------------------
if __name__ == "__main__":
    strategy_code, final_metrics, final_df = iterate_strategy_with_assistant(
        data_path="nifty50.csv",  
        assistant_strategy_generator=assistant_strategy_generator
    )
    print("\n✅ Final Strategy Code Chosen:")
    print(strategy_code)
    print("\n📊 Final Metrics:")
    print(final_metrics)
    final_df.to_csv("final_strategy_pnl.csv", index=False)
    print("\n📁 Saved to final_strategy_pnl.csv")
