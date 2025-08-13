import numpy as np # Numerical operations library for math 
import pandas as pd # Data handling library for CSV and tables 
from sklearn.linear_model import LinearRegression # Linear Regression model from Scikit-Learn

# ----------------------------------
# Step 1: Load & Clean Backend Data
# ----------------------------------
def load_backend_data(path): 
    df = pd.read_csv(path) # Load CSV data into a "DataFrame"
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]  # Remove unnamed columns (In the CSV file)

    # Convert grades to numeric, replace blanks with 0
    grade_cols = ['A', 'B', 'C', 'D', 'F', 'P', 'NP', 'IX', 'EW', 'W', 'TOTAL'] # Columns that contain the grade counts 
    df[grade_cols] = df[grade_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    return df # Return cleaned "DataFrame"

# ---------------------------------------------------------------------------------------------
# Step 2: Calculate Course-Specific Professor Difficulty (IDS | Instructor Difficulty Score)
# ---------------------------------------------------------------------------------------------
def calculate_course_specific_ids(df):
    # Compute average GPA per (Instructor, Class)
    df['MeanGPA'] = (
        4 * df['A'] + 3 * df['B'] + 2 * df['C'] + 1 * df['D'] + 0 * df['F']
    ) / (df['A'] + df['B'] + df['C'] + df['D'] + df['F'])

    # Invert GPA for difficulty (Higher GPA = Easier, subtracted from MAX GPA 4.0)
    df['Xp'] = 4.0 - df['MeanGPA']

    # Calculate mean inverted GPA per (Instructor, Class)
    difficulty_stats = df.groupby(['INSTRUCTOR', 'CLASS'])['Xp'].mean()

    # Calculate global mean and standard deviation of difficulty scores for normalization
    mu = difficulty_stats.mean()
    sigma = difficulty_stats.std()

    # Calculate Z-score normalization per (Instructor, Class)
    z_scores = (difficulty_stats - mu) / sigma

    # Define Sigmoid function to map between 0 and 1 (Scaling z-scores between 0 and 1)
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    ids = sigmoid(z_scores) # Apply sigmoid to normalized difficulty z-scores -> (IDS)

    return ids  # Series indexed by (INSTRUCTOR, CLASS)

# -----------------------------------------------
# Step 3: Calculate Student Ability Score (SAS) 
# -----------------------------------------------
def calculate_student_ability(student_history):
    # student_history: list of (course, prof, numeric_grade)
    if not student_history:
        return 2.0  # default average ability if NO HISTORY
    grades = [grade for (_, _, grade) in student_history] # Extract numeric grades only from history
    return sum(grades) / len(grades)  # simple GPA average

# -------------------------------------------
# Step 4: Build Training Data & Train Model
# -------------------------------------------
def build_training_data(student_history, prof_ids):
    rows = [] # List that hold processed student data rows 
    for course, prof, grade in student_history:
        difficulty = prof_ids.get((prof, course), 0.5)  # Get course-specific professor difficulty score OR default difficulty 0.5 if missing
        # Append dictionary for this course record
        rows.append({
            'COURSE': course,
            'INSTRUCTOR': prof,
            'PROF_DIFFICULTY': difficulty,
            'GRADE': grade
        })

    # Create DataFrame from the list of dictionaries 
    df_student = pd.DataFrame(rows)
    #Convert categorical variables (SUCH AS COURSE AND INSTUCTOR) into one-hot encoded columns
    df_encoded = pd.get_dummies(df_student, columns=['COURSE', 'INSTRUCTOR'])
    # Encodes all columns except the "target grade"
    X = df_encoded.drop('GRADE', axis=1)
    # "Target Grade" - Grade
    y = df_encoded['GRADE']
    # Returns features X, target Y, and feature column names excluding grades 
    return X, y, df_encoded.columns.drop('GRADE')

# -------------------------------
# Step 5: Predict Future Grade
# -------------------------------
def predict_future_grade(model, feature_columns, planned_course, planned_prof, prof_ids, student_ability):
    #Create an input DataFrame with one row of zeros for each feature column
    input_row = pd.DataFrame([np.zeros(len(feature_columns))], columns=feature_columns)

    # Get course-specific prof difficulty or default 0.5
    difficulty = prof_ids.get((planned_prof, planned_course), 0.5)

    # Set the professor difficulty feature for the input row 
    input_row['PROF_DIFFICULTY'] = difficulty

    # Add student ability feature 
    if 'STUDENT_ABILITY' in input_row.columns:
        input_row['STUDENT_ABILITY'] = student_ability
    else:
        # If STUDENT_ABILITY column doesn't exist, optionally blend it into difficulty 
        input_row['PROF_DIFFICULTY'] = (difficulty + student_ability / 4) / 2

    # One-hot encode COURSE and INSTRUCTOR
    course_col = f'COURSE_{planned_course}'
    prof_col = f'INSTRUCTOR_{planned_prof}'
    #Set those columns above to 1 if they exist in the feature columns
    if course_col in input_row.columns:
        input_row[course_col] = 1
    if prof_col in input_row.columns:
        input_row[prof_col] = 1
    
    # Predict the grade using the trained model and prepare the input 
    prediction = model.predict(input_row)[0]

    # Round prediction to two decimals and return
    return round(prediction, 2)

# ------------------------------------------------
# Step 6: Convert Numeric Grade to Letter Grade
# ------------------------------------------------
def grade_to_letter(g):
    return (
        'A' if g >= 3.75 else
        'B' if g >= 2.75 else
        'C' if g >= 1.75 else
        'D' if g >= 0.75 else 'F'
    )

# -------------------------------
# MAIN SCRIPT
# -------------------------------
if __name__ == "__main__":
    csv_path = 'SMC_Data.csv' # Path to CSV data file 
    backend_df = load_backend_data(csv_path) # Load and clean backend data 
    prof_ids = calculate_course_specific_ids(backend_df) # Compute course-specific professor difficulty (IDS | Instructor Difficulty Score)

    print("---- Student Input ----")
    student_history = [] # Initialize student history list 
    num_entries = int(input("Number of past courses to enter: ")) # How many past courses to input? 

    # Input past courses, professors, and grades from user
    for i in range(num_entries):
        course = input(f"Course {i+1} name: ").strip().upper()
        prof = input(f"Professor {i+1} name: ").strip().upper()
        letter = input(f"Letter Grade {i+1} (A/B/C/D/F): ").strip().upper()

        # Validate letter grade input
        while letter not in ['A', 'B', 'C', 'D', 'F']:
            letter = input("Invalid grade. Please enter A, B, C, D, or F: ").strip().upper()

        # Map letter grade to numeric scale 
        numeric_grade = {'A':4, 'B':3, 'C':2, 'D':1, 'F':0}[letter]

        # Add the course data tuple to student history list 
        student_history.append((course, prof, numeric_grade))

    # Calculate student ability score (SAS) as average numeric grade (GPA) from history 
    student_ability = calculate_student_ability(student_history)

    print("\n---- Prediction Setup ----")
    planned_course = input("Planned Course name: ").strip().upper() # Course for prediction 
    planned_prof = input("Planned Professor name: ").strip().upper() # Professor for prediction 

    X_train, y_train, _ = build_training_data(student_history, prof_ids) # Prepare training data from student history and course-specific professor difficulty scores

    # Add student ability as a feature to all training rows
    X_train['STUDENT_ABILITY'] = student_ability

    # Update feature columns after adding STUDENT_ABILITY
    feature_cols = X_train.columns

    # Create and train the linear regression model on the training data 
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict the future grade for planned ccourse & professor using the model 
    predicted_grade = predict_future_grade(model, feature_cols, planned_course, planned_prof, prof_ids, student_ability)

    # Print prediction results! 
    print("\n---- Prediction Result ----")
    print(f"Based on your history, your predicted grade for {planned_course} with {planned_prof} is:")
    print(f"➤ Letter Grade: {grade_to_letter(predicted_grade)}")
