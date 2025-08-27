// static/main.js

document.addEventListener("DOMContentLoaded", () => {
  const body = document.body;

  /* ==== THEME TOGGLE ==== */
  const toggleBtn = document.getElementById("theme-toggle");
  if (toggleBtn) {
    // Load saved theme
    if (localStorage.getItem("theme") === "dark") {
      body.classList.add("dark-mode");
      toggleBtn.textContent = "☀️ Light Mode";
    }

    toggleBtn.addEventListener("click", () => {
      body.classList.toggle("dark-mode");

      if (body.classList.contains("dark-mode")) {
        toggleBtn.textContent = "☀️ Light Mode";
        localStorage.setItem("theme", "dark");
      } else {
        toggleBtn.textContent = "🌙 Dark Mode";
        localStorage.setItem("theme", "light");
      }
    });
  }

  /* ==== ADD COURSE FORM ==== */
  const addCourseBtn = document.getElementById("add-course-btn");
  const coursesContainer = document.getElementById("courses-container");

  if (addCourseBtn && coursesContainer) {
    addCourseBtn.addEventListener("click", () => {
      const currentCourses = coursesContainer.querySelectorAll(".course-entry");
      const nextIndex = currentCourses.length + 1;

      const newCourseDiv = document.createElement("div");
      newCourseDiv.className = "course-entry";
      newCourseDiv.setAttribute("data-index", nextIndex);

      newCourseDiv.innerHTML = `
        <label>Course ${nextIndex}:</label>
        <input type="text" name="course_${nextIndex}" placeholder="Course Name_
