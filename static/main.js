// static/main.js z

document.addEventListener('DOMContentLoaded', () => {
    const addCourseBtn = document.getElementById('add-course-btn');
    const coursesContainer = document.getElementById('courses-container');

    addCourseBtn.addEventListener('click', () => {
        const currentCourses = coursesContainer.querySelectorAll('.course-entry');
        const nextIndex = currentCourses.length + 1;

        const newCourseDiv = document.createElement('div');
        newCourseDiv.className = 'course-entry';
        newCourseDiv.setAttribute('data-index', nextIndex);

        newCourseDiv.innerHTML = `
            <label>Course ${nextIndex}:</label>
            <input type="text" name="course_${nextIndex}" placeholder="Course Name" />
            <input type="text" name="professor_${nextIndex}" placeholder="Professor Name" class="professor-input" />
            <input type="text" name="grade_${nextIndex}" placeholder="Grade" class="grade-input" />
        `;

        coursesContainer.appendChild(newCourseDiv);
    });
});
