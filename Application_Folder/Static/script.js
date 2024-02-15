document.getElementById('algorithm-form').addEventListener('submit', function(event) {
    event.preventDefault();

    const algorithm = document.getElementById('algorithm').value;

    fetch('/run_algorithm', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            algorithm: algorithm,
        }),

    })
    .then(response => response.json())
    .then(data => {
        const canvas = document.getElementById('result-canvas');
        const ctx = canvas.getContext('2d');

        // Assuming data is an array of points to be plotted
        data.forEach((point) => {
            ctx.lineTo(point.x, point.y);
            ctx.stroke();
        });
    })
    .catch((error) => {
        console.error('Error:', error);
    });
});
