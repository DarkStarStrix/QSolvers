document.getElementById('algorithm-form').addEventListener('submit', function(event) {
    event.preventDefault();

    const algorithm = document.getElementById('algorithm').value;

    // Display a loading screen
    let loadingScreen = document.getElementById('loading-screen');
    let statusElement = document.getElementById('status');
    statusElement.innerText = 'Running algorithm...'
    loadingScreen.style.display = 'block';

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
        // Start polling the task status every second
        let intervalId = setInterval(() => {
            fetch(`/check_task/${data.task_id}`)
            .then(response => response.json())
            .then(data => {
                if (data.state === 'SUCCESS' || data.state === 'FAILURE') {
                    clearInterval(intervalId);

                    // Hide the loading screen
                    loadingScreen.style.display = 'none';

                    // Display the result on the canvas
                    const canvas = document.getElementById('result-canvas');
                    const ctx = canvas.getContext('2d');

                    // Assuming data.result is an array of points to be plotted
                    data.result.forEach((point) => {
                        ctx.lineTo(point.x, point.y);
                        ctx.stroke();
                    });

                    // Display the status on index.html
                    statusElement.innerText = data.status;
                }
            });
        }, 1000);
    })
    .catch((error) => {
        console.error('Error:', error);
    });
});
