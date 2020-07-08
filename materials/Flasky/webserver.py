from wsgiref.simple_server import make_server
from urllib.request import urlopen
# from configparser import ConfigParser
from functools import wraps
from flask import     (Flask,
    url_for,
    redirect,
    request,
    flash,
    render_template,
    make_response)
import json

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

def login_required(func:callable)->callable:
    @wraps(func)
    def wrapper(*args,**kwargs):
        cookie_is_set=request.cookies.get('auth')
        if cookie_is_set:
            return func(*args,**kwargs)
        else:
            flash('Login required','info')
            return redirect (url_for('login'))
    return wrapper

@app.errorhandler(404)
def not_found(err):
    return 'Nothing was found', 404


@app.errorhandler(500)
def internal_error(err):
    return 'Something went wrong', 500


# checking if searched name is available in the stored dictionary,
# and returns a list of matching names (the second returned value is a str
def name_in_list(searched_name:str,stored_data:dict):
    result = []
    for i in stored_data['person']:
        if searched_name == i['First name'] or searched_name == i["Last name"]:
            result.append(i)
    if len(result)!=0:
        output= "your search result"
    else:
        output="we couldn't find the name"
    return (result,output)

# now it is ok
@app.route("/",methods=['GET','POST'])
def post_index():
    print(url_for('static',filename='login.css'))
    data = request.form.to_dict()
    if data:
        with open('person.json') as json_file:
            persondata = json.load(json_file)
            persondata['person'].append(data)
            with open('person.json', 'w') as f:
                json.dump(persondata, f, indent=4)
    return render_template("index.html")


@app.route("/search",methods=["GET"])
def get_search():
    searched_name=(request.args.get('searched_name'))
    if searched_name:
        with open('person.json', 'r') as f:
            stored_data = json.load(f)
            persons,output=name_in_list(searched_name,stored_data)
            return render_template("search.html",persons=persons,output=output)
    return render_template("search.html")

@app.route("/login", methods=["Get"])
def login():
    return render_template("login.html",username='')

@app.route("/login", methods=["Post"])
def post_login():
    username = request.form.get('login', '').strip().lower()
    password = request.form.get('password', '').strip().lower()
    if username == 'john' and password == '123':
        flash('Login success!', 'info')
        response = make_response(redirect(url_for('dashboard')))
        response.set_cookie('auth', username)
        return response
    else:
        flash('User not found', 'error')
        return render_template('login.html',username=username)


    return render_template('login.html', username=username)


@app.route('/dashboard')
@login_required
def dashboard():
    url="http://127.0.0.1:8080/"
    with urlopen(url) as resp:
        data=resp.read().decode("utf-8")
        print(data)
        print(type(data))
    return render_template('dashboard.html',data=f"b'{data}'")


if __name__=='__main__':
    app.run(port=8080,debug=True)
