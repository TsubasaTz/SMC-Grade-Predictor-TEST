from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load and clean data once when app starts
# testing with sample  


df = pd.read_csv('SMC_Data.csv', skipinitialspace=True)

# Remove completely empty columns that come from trailing commas
df = df.dropna(axis=1, how='all')

# Replace non-breaking spaces (common in Excel exports) with NaN
df.replace('\xa0', pd.NA, regex=True, inplace=True)

grade_cols = ["A", "B", "C", "D", "F", "P", "NP", "IX", "EW", "W"]
for col in grade_cols:
 df[col] = (
        df[col]
        .astype(str)
        .str.strip()
        .replace({"": "0", "nan": "0", "NaN": "0", "<NA>": "0"})
        .fillna("0")
        .astype(int)
    )

df["Professor"] = df["INSTRUCTOR"].apply(lambda x: str(x).strip().split()[0].title())
df["C"] += df["P"]
df["F"] += df["NP"] + df["IX"]
df["W"] += df["EW"]

# Helper to create matplotlib plots as base64 img strings
def plot_to_img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str

@app.route('/')
def index():
    return redirect(url_for('prediction'))


@app.route("/predict", methods=["GET", "POST"])
def prediction():
    prediction_result = ""
    img = None

    if request.method == "POST":
        courses = request.form.getlist("course_name[]")
        instructors = request.form.getlist("instructor_name[]")
        grades = request.form.getlist("grade[]")

        for course, instructor, grade in zip(courses, instructors, grades):
            course = course.strip()
            professor = instructor.strip().title()

            subset = df[(df['Professor'] == professor) & (df['CLASS'] == course)]
            if subset.empty:
                prediction_result += f"\nCourse: {course} | Professor: {professor} ? No data found.\n"
                continue

            total_students = subset[["A", "B", "C", "D", "F"]].sum().sum()
            if total_students == 0:
                prediction_result += f"\nCourse: {course} | Professor: {professor} ? No grade data.\n"
                continue

            total_points = (
                subset["A"].sum() * 4 +
                subset["B"].sum() * 3 +
                subset["C"].sum() * 2 +
                subset["D"].sum() * 1 +
                subset["F"].sum() * 0
            )
            avg_grade_point = total_points / total_students

            if avg_grade_point >= 3.5:
                predicted = 'A'
            elif avg_grade_point >= 2.5:
                predicted = 'B'
            elif avg_grade_point >= 1.5:
                predicted = 'C'
            elif avg_grade_point >= 0.5:
                predicted = 'D'
            else:
                predicted = 'F'

            prediction_result += (
                f"\nCourse: {course} | Professor: {professor} ? "
                f"Predicted Grade: {predicted} (GPA: {avg_grade_point:.2f})\n"
            )

            # Only generate a plot for the last valid course
            labels = ['A', 'B', 'C', 'D', 'F']
            counts = [subset[g].sum() for g in labels]

            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.bar(labels, counts, color='dodgerblue')
            ax.set_title(f'Grade Distribution for {professor} in {course}')
            ax.set_ylabel('Number of Students')
            for bar, count in zip(bars, counts):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 1,
                        str(count), ha='center', color='white', fontsize=12)
            img = plot_to_img(fig)

    return render_template("prediction.html", prediction=prediction_result.strip(), plot_img=img)


@app.route("/new_student_prediction")
def new_student_prediction():
    try:
        medians = df.groupby("CLASS")[grade_cols].median().round(1)
        best_profs = (df.groupby(["CLASS", "Professor"])["A"].sum() /
                      df.groupby(["CLASS", "Professor"])["TOTAL"].sum())
        best_by_course = best_profs.groupby("CLASS").idxmax().apply(lambda x: x[1])

        fig, ax = plt.subplots(figsize=(10, 5))
        medians.T.plot(kind='bar', ax=ax)
        ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        ax.set_title("Median Grade Distribution per Course")
        ax.set_ylabel("Median Count")
        ax.set_xlabel("Grade Type")
        plt.tight_layout()

        graph_path = "static/median_prediction.png"
        plt.savefig(graph_path)
        plt.close()

        print("Best professors:", best_by_course)
        return render_template(
            "new_student_prediction.html",
            best_profs=best_by_course,
            graph=graph_path  # 
        )
    except Exception as e:
        print("Error in new_student_prediction:", e)
        return "Error in generating new student prediction. Check logs."

@app.route('/analyzer', methods=['GET', 'POST'])
def analyzer():
    analysis_result = None
    img = None

    option = request.form.get('option')
    prof_name = request.form.get('prof_name', '').title()

    if option:
        if option == '1':  # Professor Summary
            subset = df[df["Professor"] == prof_name]
            if subset.empty:
                analysis_result = "Professor not found."
            else:
                total_students = subset[["A", "B", "C", "D", "F", "W"]].sum().sum()
                details = []
                labels = ["A", "B", "C", "D", "F", "W"]
                values = []
                for g in labels:
                    count = subset[g].sum()
                    values.append(count)
                    ratio = count / total_students * 100 if total_students else 0
                    details.append(f"{g}: {count} students ({ratio:.2f}%)")
                analysis_result = f"{prof_name} Summary (total {total_students} students):\n" + "\n".join(details)

                # Plot
                fig, ax = plt.subplots(figsize=(6, 4))
                bars = ax.bar(labels, values, color="skyblue", edgecolor="black")
                ax.set_title(f"{prof_name}'s Grade Distribution")
                ax.set_ylabel("Student Count")
                ax.grid(axis="y", linestyle="--", alpha=0.5)
                for bar, count in zip(bars, values):
                    percent = (count / total_students) * 100 if total_students else 0
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 2,
                            f"{percent:.1f}%", ha='center', va='top', color='black', fontsize=9)
                img = plot_to_img(fig)

        elif option == '2':  # Average by Course
            grouped = df.groupby("CLASS")[["A", "B", "C", "D", "F", "W"]].sum()
            grouped["Total"] = grouped.sum(axis=1)
            grouped["A Ratio (%)"] = grouped["A"] / grouped["Total"] * 100
            details = grouped[["A Ratio (%)"]].to_string()
            analysis_result = "Average A ratios by course:\n" + details

            # Plot top courses
            grouped_sorted = grouped.sort_values("A Ratio (%)", ascending=False)
            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.bar(grouped_sorted.index, grouped_sorted["A Ratio (%)"], color="lightgreen", edgecolor="black")
            ax.set_xticklabels(grouped_sorted.index, rotation=45, ha="right")
            ax.set_ylabel("A Ratio (%)")
            ax.set_title("A Ratio by Course")
            ax.grid(axis="y", linestyle="--", alpha=0.5)
            for bar, ratio in zip(bars, grouped_sorted["A Ratio (%)"]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 2,
                        f"{ratio:.1f}%", ha='center', va='top', color='black', fontsize=9)
            img = plot_to_img(fig)

        elif option == '3':  # Overall Data
            labels = ["A", "B", "C", "D", "F", "W"]
            total_counts = df[labels].sum()
            total_students = total_counts.sum()
            details = []
            for g in labels:
                count = total_counts[g]
                ratio = count / total_students * 100 if total_students else 0
                details.append(f"{g}: {count} students ({ratio:.2f}%)")
            details.append(f"\nTotal students: {total_students}")
            analysis_result = "Overall Grade Distribution:\n" + "\n".join(details)

            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.bar(labels, total_counts.values, color="coral", edgecolor="black")
            ax.set_title("Overall Grade Distribution")
            ax.set_ylabel("Number of Students")
            ax.grid(axis="y", linestyle="--", alpha=0.5)
            for bar, count in zip(bars, total_counts.values):
                percent = (count / total_students) * 100 if total_students else 0
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 2,
                        f"{percent:.1f}%", ha='center', va='top', color='black', fontsize=9)
            img = plot_to_img(fig)

        elif option == '4':  # Best/Worst A Ratio
            grouped = df.groupby("Professor")[["A", "B", "C", "D", "F", "W"]].sum()
            grouped["Total"] = grouped.sum(axis=1)
            grouped["A Ratio"] = grouped["A"] / grouped["Total"] * 100
            sorted_group = grouped[grouped["Total"] > 0].sort_values(by="A Ratio", ascending=False)
            best = sorted_group.head(1)
            worst = sorted_group.tail(1)
            details = "\nBest Professor:\n" + best[["A", "Total", "A Ratio"]].to_string()
            details += "\n\nWorst Professor:\n" + worst[["A", "Total", "A Ratio"]].to_string()
            analysis_result = details

        elif option == '5':  # Full A Ratio Ranking
            grouped = df.groupby("Professor")[["A", "B", "C", "D", "F", "W"]].sum()
            grouped["Total"] = grouped.sum(axis=1)
            grouped = grouped[grouped["Total"] > 0]
            grouped["A Ratio (%)"] = grouped["A"] / grouped["Total"] * 100
            ranked = grouped.sort_values(by="A Ratio (%)", ascending=False)[["A Ratio (%)", "Total"]]
            analysis_result = "Full Professor A Ratio Ranking:\n" + ranked.to_string()

            # Plot top 10
            top_n = ranked.head(10)
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(top_n.index, top_n["A Ratio (%)"], color="slateblue", edgecolor="black")
            ax.set_xticklabels(top_n.index, rotation=45, ha="right")
            ax.set_ylabel("A Ratio (%)")
            ax.set_title("Top 10 Professors by A Ratio")
            ax.grid(axis="y", linestyle="--", alpha=0.5)
            for bar, ratio in zip(bars, top_n["A Ratio (%)"]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 2,
                        f"{ratio:.1f}%", ha='center', va='top', color='white', fontsize=9)
            img = plot_to_img(fig)

    return render_template('analyzer.html', analysis=analysis_result, plot_img=img)
# ... existing route functions ...

@app.route("/fall_classes", methods=["GET", "POST"])
def fall_classes():
    results = []
    class_name = ""
    if request.method == "POST":
        class_name = request.form.get("class_name", "").replace(" ", "").upper()
        course_df = pd.read_csv("SMC_Math_Course(Sheet1).csv")
        course_df["Course Name Norm"] = course_df["Course Name"].str.replace(" ", "").str.upper()
        matches = course_df[course_df["Course Name Norm"] == class_name]
        for idx, row in matches.iterrows():
            prof_last = str(row['Instructor']).strip().split()[0].title()
            prof_data = df[df["Professor"] == prof_last]
            total = prof_data[["A", "B", "C", "D", "F", "W"]].sum().sum()
            a_count = prof_data["A"].sum()
            a_ratio = round((a_count / total * 100), 2) if total else 0
            results.append({
                "section": row['Section'],
                "title": row['Course Title'],
                "schedule": row['Schedule'],
                "modality": row['Section Modality'],
                "campus": row['Campus'],
                "location": row['Location'],
                "instructor": row['Instructor'],
                "a_ratio": a_ratio
            })
    return render_template("fall_classes.html", results=results, class_name=class_name)

# ... then your main block ...
if __name__ == '__main__':
    app.run(debug=True)

