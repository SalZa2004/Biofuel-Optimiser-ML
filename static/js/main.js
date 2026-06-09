document.addEventListener("DOMContentLoaded", function () {
    const wrapper = document.getElementById("wrapper");
    const sidebar = document.getElementById("sidebar-wrapper");
    const toggleBtn = document.getElementById("sidebarToggle");

    toggleBtn.addEventListener("click", function () {
        // toggle hidden state
        sidebar.classList.toggle("collapsed");
    });
});
