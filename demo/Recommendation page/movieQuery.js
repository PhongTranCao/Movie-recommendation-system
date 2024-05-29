const API_Key = '&api_key=156c9d1dd5f65606e4645a7b9a62644a'
const base_URL = 'https://api.themoviedb.org/3/search/movie?query='
const image_URL = 'https://image.tmdb.org/t/p/w500'
const apiKey = 'AIzaSyDJ6-xvW5vxyDCkf6SoiUOfay2Wlne_m-4'
const maxResults = 1;
const main = document.getElementById('carousel-inner')
let movies;
// const test_URL = 'https://api.themoviedb.org/3/search/movie?include_adult=false&query=anime&page=1&sort_by=popularity.desc&' + API_Key

// Initial Movies Data
fetch('../../data_processing/movies_recommend_name_list.json')
    .then(response => { return response.json()})
    .then(data => movies = data)
// End Initial Movies Data


function processMovieLabel(label) {
    return label.replace(/ /g, '+').replace(/'/g, '%27')
}

function getMovies() {
    fetch('../../data_processing/movies_recommend_name_list.json')
        .then(response => response.json())
        .then(movies => {
            const data_for_shown = [];
            const fetchPromises = movies.map(movie => {
                const name = processMovieLabel(movie.label);
                const url = `${base_URL}${name}${API_Key}`;

                return fetch(url)
                    .then(res => res.json())
                    .then(data => {
                        if (data.results && data.results[0]) data_for_shown.push(data.results[0]);
                    });});

            Promise.all(fetchPromises)
                .then(() => {
                    console.log(data_for_shown);
                    showMovies(data_for_shown);
                })
                .catch(error => {
                    console.error('Error fetching movies:', error);
                });
        })
        .catch(error => {
            console.error('Error loading json:', error);
        });
}

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

// create carousel from data
function showMovies (data) {
        main.innerHTML = '';
        let count = 0;
        let activeOrNot = 'active';
        data.forEach(movie => {
            const {original_title, poster_path, vote_average, overview} = movie;
            const movieEl = document.createElement('div');
            movieEl.classList.add('carousel-item-wrapper');
            movieEl.innerHTML = `
            <div class="carousel-item ${activeOrNot}" data-bs-interval="4000" style="background-image: url(${poster_path? image_URL+poster_path: "http://via.placeholder.com/1080x1580"})">
                <div class="hero-slide-item-content">
                    <img src="${poster_path? image_URL+poster_path: "http://via.placeholder.com/1080x1580"}" alt="" style="z-index: 1000;">
                    <div class="item-content-wrapper">
                        <div class="item-content-title top-down mb-1 animate__animated animate__fadeInDown animate__delay-halfasec">${original_title}</div>
                        <div class="movie-infos">
                            <div class="movie-info">
                                <i class="fa fa-star animate__animated animate__fadeInLeft"></i>
                                <span class="animate__animated animate__fadeInLeft animate__delay-halfasec">${vote_average}</span>
                            </div>
                        </div>
                        <div class="item-content-description animate__animated animate__fadeInLeft">${overview}</div>
                        <div class="item-action mt-3 animate__animated animate__fadeInLeft">
                            <a href="#" id="show-video-btn" class="btn btn-hover" data-title="${original_title}">
                                <span>Watch now</span>
                            </a>
                        </div>
                    </div>
                </div>
            </div>
            `
            main.appendChild(movieEl);
            if (count === 0){
                count = 1;
                activeOrNot = ' ';
            }
        });
        addClickEventToLinks();
}


// autocomplete for search-box
const input = document.getElementById("search-box");
autocomplete({
    input: input,
    fetch: function(text, update) {
        text = text.toLowerCase();
        // you can also use AJAX requests instead of preloaded data
        let suggestions = movies.filter(n => n.label.toLowerCase().startsWith(text))
        update(suggestions);
    },
    onSelect: function(item) {
        input.value = item.label;
    }
});


function sendData() {
    const value = document.getElementById('search-box').value;
    $.ajax({
        url: 'http://127.0.0.1:5000/process',
        type: 'POST',
        data: { 'data': value },
        success: function() {
            getMovies()
            // document.getElementById('output').innerHTML = response;
        },
        error: function(error) {
            console.log(error);
        }
    });
}


