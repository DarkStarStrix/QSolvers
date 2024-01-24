document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('userTypeForm').addEventListener('submit', function(event) {
        event.preventDefault();
        var isUser = document.getElementById('user').checked;
        var isBusiness = document.getElementById('business').checked;
        if (isUser) {
            window.location.href = 'user.html';
        } else if (isBusiness) {
            window.location.href = 'business.html';
        }
    });
});
