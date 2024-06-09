document.getElementById('video-overlay').addEventListener('click', (event) => {
    if (event.target.id === 'video-overlay') {
        const overlay = document.getElementById('video-overlay');
        overlay.classList.remove('show');
        setTimeout(() => {
            overlay.style.display = 'none';
            const videoContent = document.getElementById('video-content');
            videoContent.innerHTML = '';
        }, 500);
    }
});

async function fetchVideo(query) {
    const url = `https://www.googleapis.com/youtube/v3/search?part=snippet&q=${encodeURIComponent(query)}&key=${apiKey}&maxResults=${maxResults}&type=video`;

    try {
        const response = await fetch(url);
        const data = await response.json();
        const videoId = data.items[0].id.videoId;
        const videoUrl = `https://www.youtube.com/embed/${videoId}`;

        const videoContent = document.getElementById('video-content');
        videoContent.innerHTML = `<iframe width="910" height="500" src="${videoUrl}" allow="accelerometer; picture-in-picture; fullscreen"></iframe>`;
    } catch (error) {
        console.error('Error fetching video:', error);
    }
}

function addClickEventToLinks() {
    const links = document.querySelectorAll('#show-video-btn');
    links.forEach(link => {
        link.addEventListener('click', (event) => {
            event.preventDefault();
            const query = link.getAttribute('data-title');
            const overlay = document.getElementById('video-overlay');
            overlay.style.display = 'flex';
            setTimeout(() => {
                overlay.classList.add('show');
                }, 10);
            fetchVideo(query);
        });
    });
}