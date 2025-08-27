from flask import Flask, render_template, request
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Required for Flask, avoid GUI backend
import matplotlib.pyplot as plt
import io, base64

app = Flask(__name__)

# Load & clean data once
df = pd.read_csv("SMC_Data.csv")

grade_cols = ["A", "B", "C", "D", "F", "P", "NP", "IX", "EW", "W"]
for col in grade_cols:
    df[col] = (
        df[col]
        .astype(str)
        .str.strip()
        .replace({"": "0", "nan": "0", "NaN": "0"})
        .fillna("0")
        .astype(int)
    )

# Normalize professor name + grades
df["Professor"] = df["INSTRUCTOR"].apply(lambda x: str(x).strip().split()[0].title())
df["C"] += df["P"]
df["F"] += df["NP"] + df["IX"]
df["W"] += df["EW"]

# Helper to convert matplotlib figure → base64
def fig_to_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


@app.route("/", methods=["GET", "POST"])
def index():
    analysis, plot_img = None, None

    if request.method == "POST":
        choice = request.form.get("choice")

        # -------- Option 1: Professor Summary --------
        if choice == "1":
            name = request.form.get("prof_name", "").title()
            subset = df[df["Professor"] == name]

            if subset.empty:
                analysis = f"Professor {name} not found."
            else:
                total_students = subset[["A", "B", "C", "D", "F", "W"]].sum().sum()
                lines, values, labels = [], [], ["A", "B", "C", "D", "F", "W"]

                for g in labels:
                    count = subset[g].sum()
                    values.append(count)
                    ratio = count / total_students * 100 if total_students else 0
                    lines.append(f"{g}: {count} students ({ratio:.2f}%)")

                analysis = f"{name} Summary (total {total_students} students):\n" + "\n".join(
                    lines
                )

                plt.figure(figsize=(6, 4))
                bars = plt.bar(labels, values, color="skyblue", edgecolor="black")
                plt.title(f"{name}'s Grade Distribution")
                plt.ylabel("Student Count")
                plt.grid(axis="y", linestyle="--", alpha=0.5)
                for bar, count in zip(bars, values):
                    pct = (count / total_students) * 100
                    plt.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() - 2,
                        f"{pct:.1f}%",
                        ha="center",
                        va="top",
                        color="black",
                        fontsize=9,
                    )
                plt.tight_layout()
                plot_img = fig_to_base64()
                plt.close()

        # -------- Option 2: Average by Course --------
        elif choice == "2":
            grouped = df.groupby("CLASS")[["A", "B", "C", "D", "F", "W"]].sum()
            grouped["Total"] = grouped.sum(axis=1)
            grouped["A Ratio (%)"] = grouped["A"] / grouped["Total"] * 100
            analysis = grouped[["A Ratio (%)"]].to_string()

            grouped_sorted = grouped.sort_values("A Ratio (%)", ascending=False)
            plt.figure(figsize=(10, 5))
            bars = plt.bar(
                grouped_sorted.index,
                grouped_sorted["A Ratio (%)"],
                color="lightgreen",
                edgecolor="black",
            )
            plt.xticks(rotation=45, ha="right")
            plt.ylabel("A Ratio (%)")
            plt.title("A Ratio by Course")
            plt.grid(axis="y", linestyle="--", alpha=0.5)
            plot_img = fig_to_base64()
            plt.close()

        # -------- Option 3: Overall Data --------
        elif choice == "3":
            labels = ["A", "B", "C", "D", "F", "W"]
            total_counts = df[labels].sum()
            total_students = total_counts.sum()
            lines, values = [], []

            for g in labels:
                count = total_counts[g]
                values.append(count)
                ratio = count / total_students * 100 if total_students else 0
                lines.append(f"{g}: {count} students ({ratio:.2f}%)")

            analysis = "\n".join(lines) + f"\n\nTotal students: {total_students}"

            plt.figure(figsize=(6, 4))
            plt.bar(labels, values, color="coral", edgecolor="black")
            plt.title("Overall Grade Distribution")
            plt.ylabel("Number of Students")
            plt.grid(axis="y", linestyle="--", alpha=0.5)
            plot_img = fig_to_base64()
            plt.close()

        # -------- Option 4: Best/Worst Professors --------
        elif choice == "4":
            grouped = df.groupby("Professor")[["A", "B", "C", "D", "F", "W"]].sum()
            grouped["Total"] = grouped.sum(axis=1)
            grouped["A Ratio"] = grouped["A"] / grouped["Total"] * 100
            sorted_group = grouped[grouped["Total"] > 0].sort_values(
                by="A Ratio", ascending=False
            )
            best, worst = sorted_group.head(1), sorted_group.tail(1)
            analysis = f"Best Professor:\n{best[['A','Total','A Ratio']]}\n\nWorst Professor:\n{worst[['A','Total','A Ratio']]}"

        # -------- Option 5: Full Professor A Ratio Ranking --------
        elif choice == "5":
            grouped = df.groupby("Professor")[["A", "B", "C", "D", "F", "W"]].sum()
            grouped["Total"] = grouped.sum(axis=1)
            grouped = grouped[grouped["Total"] > 0]
            grouped["A Ratio (%)"] = grouped["A"] / grouped["Total"] * 100
            ranked = grouped.sort_values(by="A Ratio (%)", ascending=False)[
                ["A Ratio (%)", "Total"]
            ]
            analysis = ranked.to_string()

            top_n = ranked.head(10)
            plt.figure(figsize=(10, 6))
            plt.bar(
                top_n.index, top_n["A Ratio (%)"], color="slateblue", edgecolor="black"
            )
            plt.xticks(rotation=45, ha="right")
            plt.ylabel("A Ratio (%)")
            plt.title("Top 10 Professors by A Ratio")
            plt.grid(axis="y", linestyle="--", alpha=0.5)
            plot_img = fig_to_base64()
            plt.close()

        # -------- Option 6: Fall 2025 Courses --------
        elif choice == "6":
            class_name = request.form.get("class_name", "").replace(" ", "").upper()
            course_df = pd.read_csv("SMC_Math_Course(Sheet1).csv")
            course_df["Course Name Norm"] = (
                course_df["Course Name"].str.replace(" ", "").str.upper()
            )
            matches = course_df[course_df["Course Name Norm"] == class_name]

            if matches.empty:
                analysis = f"No matching classes found for {class_name}."
            else:
                lines = []
                for _, row in matches.iterrows():
                    prof_last = str(row["Instructor"]).strip().split()[0].title()
                    prof_data = df[df["Professor"] == prof_last]
                    total = prof_data[["A", "B", "C", "D", "F", "W"]].sum().sum()
                    a_count = prof_data["A"].sum()
                    a_ratio = (a_count / total * 100) if total else 0
                    lines.append(
                        f"Section: {row['Section']}, Title: {row['Course Title']}, "
                        f"Schedule: {row['Schedule']}, Modality: {row['Section Modality']}, "
                        f"Campus: {row['Campus']}, Location: {row['Location']}, "
                        f"Instructor: {row['Instructor']}, A Ratio: {a_ratio:.2f}%"
                    )
                analysis = "\n".join(lines)

        # -------- Option 7: Exit --------
        elif choice == "7":
            analysis = "Exiting (disabled in web version)."

        else:
            analysis = "Invalid choice."

    return render_template("index.html", analysis=analysis, plot_img=plot_img)


if __name__ == "__main__":
    app.run(debug=True)
