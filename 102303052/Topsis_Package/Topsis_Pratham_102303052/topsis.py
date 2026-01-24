import sys
import os
import pandas as pd
import numpy as np

def main():
    if len(sys.argv) != 5:
        print("Error: Wrong number of arguments.")
        print("Usage: python <program.py> <InputDataFile> <Weights> <Impacts> <ResultFileName>")
        print('Example: python 101556.py 101556-data.csv "1,1,1,2" "+,+,-,+" 101556-result.csv')
        sys.exit(1)

    input_file = sys.argv[1]
    weights_str = sys.argv[2]
    impacts_str = sys.argv[3]
    result_file = sys.argv[4]

    if not os.path.isfile(input_file):
        print("Error: File not found.")
        sys.exit(1)

    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"Error: Could not read file. {e}")
        sys.exit(1)

    if len(df.columns) < 3:
        print("Error: Input file must contain three or more columns.")
        sys.exit(1)

    try:
        temp_df = df.iloc[:, 1:].values.astype(float)
    except ValueError:
        print("Error: From 2nd to last columns must contain numeric values only.")
        sys.exit(1)

    try:
        weights = [float(w) for w in weights_str.split(',')]
    except ValueError:
        print("Error: Weights must be numeric values separated by commas.")
        sys.exit(1)

    impacts = impacts_str.split(',')

    if not all(i in ['+', '-'] for i in impacts):
        print("Error: Impacts must be either + or -.")
        sys.exit(1)

    num_cols = temp_df.shape[1]
    if len(weights) != num_cols or len(impacts) != num_cols:
        print("Error: Number of weights, number of impacts and number of columns (from 2nd to last) must be same.")
        sys.exit(1)

    rss = np.sqrt(np.sum(temp_df**2, axis=0))
    normalized_matrix = temp_df / rss

    weighted_matrix = normalized_matrix * weights

    ideal_best = []
    ideal_worst = []

    for i in range(num_cols):
        if impacts[i] == '+':
            ideal_best.append(np.max(weighted_matrix[:, i]))
            ideal_worst.append(np.min(weighted_matrix[:, i]))
        else:
            ideal_best.append(np.min(weighted_matrix[:, i]))
            ideal_worst.append(np.max(weighted_matrix[:, i]))

    ideal_best = np.array(ideal_best)
    ideal_worst = np.array(ideal_worst)

    s_plus = np.sqrt(np.sum((weighted_matrix - ideal_best)**2, axis=1))
    s_minus = np.sqrt(np.sum((weighted_matrix - ideal_worst)**2, axis=1))

    performance_score = s_minus / (s_plus + s_minus)

    df['Topsis Score'] = performance_score

    df['Rank'] = df['Topsis Score'].rank(ascending=False).astype(int)

    try:
        df.to_csv(result_file, index=False)
        print(f"Success! Result saved to {result_file}")
    except Exception as e:
        print(f"Error saving file: {e}")

if __name__ == "__main__":
    main()