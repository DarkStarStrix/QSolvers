document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('userTypeForm').addEventListener('submit', function(event) {
        event.preventDefault();
        const isUser = document.getElementById('user').checked;
        const isBusiness = document.getElementById('business').checked;
        if (isUser) {
            window.location.href = 'user.html';
        } else if (isBusiness) {
            window.location.href = 'business.html';
        }
    });
});
