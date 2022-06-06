from recipe_search.search_engine.search_module import search_module
from recipe_search.search_engine.recommend_module import recommend_module
from flask import Flask, request, render_template

app = Flask(__name__)

search_engine = search_module()
recommend_engine = recommend_module(search_engine)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/search')
def search():
    query = request.args.get('query')
    if query == '':
        return no_results_template(query)
    search_results = search_engine.search(query=query)
    if not search_results:
        return no_results_template(query)
    recommend_engine.add_user_history(search_results[0])
    return render_template('search_results.html',
                           search_results=search_results,
                           query=query)


@app.route('/search/lucky')
def search_lucky():
    query = request.args.get('query')
    recommend_results = recommend_engine.recommend()
    if not recommend_results:
        return no_results_template('recommend')
    return render_template('recommend_results.html',
                           recommend_results=recommend_results)


def no_results_template(query):
    return render_template('simple_message.html',
                           title='No results found',
                           message='Your search - <b>' + query +
                           '</b> - did not match any documents.'
                           '<br>Suggestions:<br><ul>'
                           '<li>Make sure that all words are spelled correctly.</li>'
                           '<li>Try different keywords.</li>'
                           '<li>Try more general keywords.</li>'
                           '<li>Try fewer keywords.</ul>'
                           )
