//TMDB 
const API_Key = '&api_key=156c9d1dd5f65606e4645a7b9a62644a'
const base_URL = 'https://api.themoviedb.org/3/search/movie?query='
const API_KEY = 'api_key=1cf50e6248dc270629e802686245c2c8';
const BASE_URL = 'https://api.themoviedb.org/3';
const API_URL = BASE_URL + '/discover/movie?'+API_KEY;
const IMG_URL = 'https://image.tmdb.org/t/p/w500';
const searchURL = BASE_URL + '/search/movie?'+API_KEY;
let genres, movies;

fetch('../../data_processing/movies.json')
    .then(response => { return response.json()})
    .then(data => movies = data)

fetch('../../data_processing/tags_id.json')
    .then(response => { return response.json()})
    .then(data => {
        genres = data
        Promise.all(genres).then(()=> {
            console.log(genres);
            setGenre();})
    })

const main = document.getElementById('main');
const form =  document.getElementById('form');
const search = document.getElementById('search');
const tagsEl = document.getElementById('tags');

const prev = document.getElementById('prev')
const next = document.getElementById('next')
const current = document.getElementById('current')

var currentPage = 1;
var nextPage = 2;
var prevPage = 3;
var lastUrl = '';
var totalPages = 100;

var selectedGenre = []
setGenre();
function setGenre() {
    tagsEl.innerHTML= '';
    genres.forEach(genre => {
        const t = document.createElement('div');
        t.classList.add('tag');
        t.id=genre.id;
        t.innerText = genre.tag;
        t.addEventListener('click', () => {
            if(selectedGenre.length == 0){
                selectedGenre.push(genre.id);
            }else{
                if(selectedGenre.includes(genre.id)){
                    selectedGenre.forEach((id, idx) => {
                        if(id == genre.id){
                            selectedGenre.splice(idx, 1);
                        }
                    })
                }else{
                    selectedGenre.push(genre.id);
                }
            }
            console.log(selectedGenre)
            getMovies(/*API_URL + '&with_genres='+encodeURI(selectedGenre.join(','))*/)
            highlightSelection()
        })
        tagsEl.append(t);
    })
}

function highlightSelection() {
    const tags = document.querySelectorAll('.tag');
    tags.forEach(tag => {
        tag.classList.remove('highlight')
    })
    clearBtn();
    if(selectedGenre.length !=0){
        selectedGenre.forEach(id => {
            const hightlightedTag = document.getElementById(id);
            hightlightedTag.classList.add('highlight');
        })
    }

}

function clearBtn() {
    let clearBtn = document.getElementById('clear');
    if (clearBtn) {
        clearBtn.classList.add('highlight')
    } else {
        let clear = document.createElement('div');
        clear.classList.add('tag', 'highlight');
        clear.id = 'clear';
        clear.innerText = 'Clear x';
        clear.addEventListener('click', () => {
            selectedGenre = [];
            setGenre();
            clearMovies(); // Xóa danh sách phim khi người dùng xóa thể loại
            getMovies();
        });
        tagsEl.append(clear);
    }
}

getMovies();

function isGenreSelected() {
    return selectedGenre.length > 0;
}

function processMovieLabel(label) {
    return label.replace(/ /g, '+').replace(/'/g, '%27')
}

function getMovies() {
    if (isGenreSelected()) {
        fetch('../../data_processing/movies_recommend_name_list.json')
            .then(res => res.json())
            .then(data => {
                const data_for_shown = [];
                const fetchPromises = data.map(movie => {
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
                    if (!data_for_shown.results) {
                    showMovies(data_for_shown);

                    tagsEl.scrollIntoView({ behavior: 'smooth' })
                } else {
                    clearMovies(); // Nếu không có kết quả, xóa danh sách phim
                    main.innerHTML = `<h1 class="no-results">No Results Found</h1>`;
                }
                })
            });
    } else {
        clearMovies(); // Nếu chưa chọn thể loại, xóa danh sách phim
    }
}

function clearMovies() {
    main.innerHTML = '';
}



function showMovies(data) {
    main.innerHTML = '';

    const filteredMovies = movies.filter(movie => selectedGenre.every(tag => movie.tags.includes(tag)));
    const slicedData = data.filter(movie =>
        filteredMovies.some(filteredMovie => filteredMovie.title === movie.movie_title)
    );
    slicedData.forEach(movie => {
        const { title, poster_path, vote_average, overview, id } = movie;
        const movieEl = document.createElement('div');
        movieEl.classList.add('movie');
        movieEl.innerHTML = `
             <img src="${poster_path ? IMG_URL + poster_path : "http://via.placeholder.com/1080x1580"}" alt="${title}">
            <div class="movie-info">
                <h3>${title}</h3>
                <span class="${getColor(vote_average)}">${vote_average}</span>
            </div>
            <div class="overview">
                <h3>Overview</h3>
                ${overview}
                <br/> 
                <button class="know-more" id="${title}">Watch Now</button>
            </div>
        `;
        main.appendChild(movieEl);

        document.getElementById(title).addEventListener('click', () => {
            localStorage.setItem('movieTitle', title);
            window.location.href = '../Recommendation page/carousel.html';
        });
    });
}

function getColor(vote) {
    if(vote>= 8){
        return 'green'
    }else if(vote >= 5){
        return "orange"
    }else{
        return 'red'
    }
}

form.addEventListener('submit', (e) => {
    e.preventDefault();

    const searchTerm = search.value;
    selectedGenre=[];
    setGenre();
    if(searchTerm) {
        getMovies(searchURL+'&query='+searchTerm)
    }else{
        getMovies(API_URL);
    }

})



