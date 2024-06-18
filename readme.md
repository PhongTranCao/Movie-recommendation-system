# Intro to AI (IT3160E) Project 

## Movie Recommendation System - MovieGems

### Introduction
In this project, we implements three models for a movie recommendation system: FunkSVD, Item-based Collaborative Filtering with Cosine Similarity, and Content-based Filtering with Ridge Regression. Using a range of machine learning and deep learning techniques, the project evaluates the effectiveness of each model. The goal is to identify the strengths and weaknesses of these models to determine the most suitable one for the recommendation system, thereby improving the user experience on a website's interface.

### Project Organization
```
model/                        # model deployment
...
README.md        
```
### Dataset
MovieLens1M from GroupLens
### Installation
- Install python, latest version recommended
- Install packages/libraries: pandas, numpy, scikit-learn, numba, flask, flask-cors, pickle
- Browser with latest version recommended, in order to fully support web's features.
### Usage
- For evaluation, run "ibcf/RMSE.py" or "FunkSVD/"
- For running the web
  - Firstly, start flask server ("FunkSVD/flask_funksvd.py" or "ibcf/flask_ibcf.py")
  - Choose the page you want in Web folders and start it
  - You can watch the process in the flask server file and turn on the developer mode of the website in order to see the actual movies queried
### Collaborators
<table>
    <tbody>
        <tr>
            <th align="center">Member name</th>
            <th align="center">Student ID</th>
        </tr>
        <tr>
            <td>Đặng Văn Nhân</td>
            <td align="center"> 20225990&nbsp;&nbsp;&nbsp;</td>
        </tr>
        <tr>
            <td>Nguyen Lan Nhi</td>
            <td align="center"> 20225991&nbsp;&nbsp;&nbsp;</td>
        </tr>
        <tr>
            <td>Trần Cao Phong</td>
            <td align="center"> 20226061&nbsp;&nbsp;&nbsp;</td>
        </tr>
        <tr>
            <td>Bùi Hà My</td>
            <td align="center"> 20225987&nbsp;&nbsp;&nbsp;</td>
        </tr>
        <tr>
            <td>Trần Kim Cương</td>
            <td align="center"> 20226017&nbsp;&nbsp;&nbsp;</td>
        </tr>
    </tbody>
</table>
