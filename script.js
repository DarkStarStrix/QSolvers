document.getElementById('algorithm-form').addEventListener('submit', function(event) {
    event.preventDefault();

    const algorithm = document.getElementById('algorithm').value;
    const parameters = document.getElementById('parameters').value;

    fetch('/run_algorithm', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            algorithm: algorithm,
            parameters: parameters,
        }),
    })
    .then(response => response.text())
    .then(data => {
        document.getElementById('result').textContent = data;
    })
    .catch((error) => {
        console.error('Error:', error);
    });
});
