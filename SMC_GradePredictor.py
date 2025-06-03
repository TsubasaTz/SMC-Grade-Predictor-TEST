from random import choice
import pandas as pd
import os
import matplotlib.pyplot as plt



# Load and clean data
df = pd.read_csv("SMC_Data.csv")

grade_cols = ["A", "B", "C", "D", "F", "P", "NP", "IX", "EW", "W"]
for col in grade_cols:
    df[col] = df[col].astype(str).str.strip().replace({"": "0", "nan": "0", "NaN": "0"}).fillna("0").astype(int)

# Add Professor (last name only)
df["Professor"] = df["INSTRUCTOR"].apply(lambda x: str(x).strip().split()[0].title())

# Normalize grades
df["C"] += df["P"]
df["F"] += df["NP"] + df["IX"]
df["W"] += df["EW"]

# Main interactive loop
while True:
    print("\n--- Grade Analyzer ---")
    print("1. Professor Summary")
    print("2. Average by Course")
    print("3. Overall Data")
    print("4. Best/Worst A Ratio (Professor)")
    print("5. Full A Ratio Ranking")
    print("6. Exit")
    choice = input("Select option (1-6): ")

    if choice == "1":
        name = input("Enter professor's last name: ").title()
        subset = df[df["Professor"] == name]
        if subset.empty:
            print("Professor not found.")
            continue

        total_students = subset[["A", "B", "C", "D", "F", "W"]].sum().sum()
        print(f"\n{name} Summary (total {total_students} students):")
        values = []
        labels = ["A", "B", "C", "D", "F", "W"]
        for g in labels:
            count = subset[g].sum()
            values.append(count)
            ratio = count / total_students * 100 if total_students else 0
            print(f"{g}: {count} students ({ratio:.2f}%)")

        # Visualization
        plt.figure(figsize=(6, 4))
        bars = plt.bar(labels, values, color="skyblue", edgecolor="black")
        plt.title(f"{name}'s Grade Distribution")
        plt.ylabel("Student Count")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        for bar, count in zip(bars, values):
            percent = (count / total_students) * 100
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 2,
                    f"{percent:.1f}%", ha='center', va='top', color='black', fontsize=9)
        plt.tight_layout()
        plt.show()



    elif choice == "2":
        print("\nAverage A ratios by course:")
        grouped = df.groupby("CLASS")[["A", "B", "C", "D", "F", "W"]].sum()
        grouped["Total"] = grouped.sum(axis=1)
        grouped["A Ratio (%)"] = grouped["A"] / grouped["Total"] * 100
        print(grouped[["A Ratio (%)"]])

        # Visualization
        grouped_sorted = grouped.sort_values("A Ratio (%)", ascending=False)
        plt.figure(figsize=(10, 5))
        bars = plt.bar(grouped_sorted.index, grouped_sorted["A Ratio (%)"], color="lightgreen", edgecolor="black")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("A Ratio (%)")
        plt.title("A Ratio by Course")
        plt.grid(axis="y", linestyle="--", alpha=0.5)

        # Add percentages inside bars
        for bar, ratio in zip(bars, grouped_sorted["A Ratio (%)"]):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 2,
                     f"{ratio:.1f}%", ha='center', va='top', color='black', fontsize=9)

        plt.tight_layout()
        plt.show()



    elif choice == "3":
        print("\nOverall Grade Distribution:")
        labels = ["A", "B", "C", "D", "F", "W"]
        total_counts = df[labels].sum()
        total_students = total_counts.sum()

        values = []
        for g in labels:
            count = total_counts[g]
            values.append(count)
            ratio = count / total_students * 100 if total_students else 0
            print(f"{g}: {count} students ({ratio:.2f}%)")

        print(f"\nTotal students: {total_students}")

        # Visualization
        plt.figure(figsize=(6, 4))
        bars = plt.bar(labels, values, color="coral", edgecolor="black")
        plt.title("Overall Grade Distribution")
        plt.ylabel("Number of Students")
        plt.grid(axis="y", linestyle="--", alpha=0.5)

        # Add percentages inside bars
        for bar, count in zip(bars, values):
            percent = (count / total_students) * 100
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 2,
                     f"{percent:.1f}%", ha='center', va='top', color='black', fontsize=9)

        plt.tight_layout()
        plt.show()



    elif choice == "4":
        print("\nBest and Worst Professors by A Ratio:")

        grouped = df.groupby("Professor")[["A", "B", "C", "D", "F", "W"]].sum()
        grouped["Total"] = grouped.sum(axis=1)
        grouped["A Ratio"] = grouped["A"] / grouped["Total"] * 100

        sorted_group = grouped[grouped["Total"] > 0].sort_values(by="A Ratio", ascending=False)

        best = sorted_group.head(1)
        worst = sorted_group.tail(1)

        print("\n Best Professor:")
        print(best[["A", "Total", "A Ratio"]])

        print("\n Worst Professor:")
        print(worst[["A", "Total", "A Ratio"]])
     
    elif choice == "5":
        print("\n Full Professor A Ratio Ranking:")

        grouped = df.groupby("Professor")[["A", "B", "C", "D", "F", "W"]].sum()
        grouped["Total"] = grouped.sum(axis=1)
        grouped = grouped[grouped["Total"] > 0]  # Avoid divide-by-zero
        grouped["A Ratio (%)"] = grouped["A"] / grouped["Total"] * 100

        ranked = grouped.sort_values(by="A Ratio (%)", ascending=False)
        ranked = ranked[["A Ratio (%)", "Total"]]

        print(ranked.to_string(index=True))

        # Visualization: Top 10 professors
        top_n = ranked.head(10)
        plt.figure(figsize=(10, 6))
        bars = plt.bar(top_n.index, top_n["A Ratio (%)"], color="slateblue", edgecolor="black")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("A Ratio (%)")
        plt.title("Top 10 Professors by A Ratio")
        plt.grid(axis="y", linestyle="--", alpha=0.5)

        # Add percentages inside bars
        for bar, ratio in zip(bars, top_n["A Ratio (%)"]):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 2,
                     f"{ratio:.1f}%", ha='center', va='top', color='white', fontsize=9)

        plt.tight_layout()
        plt.show()



    elif choice == "6":
        print("Exiting.")
        break

    else:
        print("Invalid choice.")
