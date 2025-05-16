from flask import Flask, render_template, request
import os 

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    # Here you can grab form data and process it
    department = request.form.get('department')
    courses = []
    for i in range(1, 3):  # because we made 2 course boxes for now
        course = request.form.get(f'course_{i}')
        professor = request.form.get(f'professor_{i}')
        grade = request.form.get(f'grade_{i}')
        if course or professor or grade:
            courses.append({
                'course': course,
                'professor': professor,
                'grade': grade
            })
    # For now just print to console
    print('Department:', department)
    print('Courses:', courses)
    return "Form submitted! Check your terminal."

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
