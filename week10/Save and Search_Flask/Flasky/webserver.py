from wsgiref.simple_server import make_server
# from configparser import ConfigParser
from flask import Flask,render_template,request,jsonify
import json

app = Flask(__name__)

# checking if searched name is available in the stored dictionary,
# and returns a list of matching names (the second returned value is a str
def name_in_list(searched_name:str,stored_data:dict)->str:
    result = []
    for i in stored_data['person']:
        if searched_name == i['First name'] or searched_name == i["Last name"]:
            result.append(i)
    if len(result)!=0:
        output= "your search result"
    else:
        output="we couldn't find the name"
    return (result,output)


@app.route("/",methods=['GET','POST'])
def post_index():
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
    searched_name=(request.args.getlist('searched_name'))
    if searched_name:
        with open('person.json', 'r') as f:
            stored_data = json.load(f)
        persons=name_in_list(searched_name[0],stored_data)[0]
        output=name_in_list(searched_name[0],stored_data)[1]
        return render_template("search.html",persons=persons,output=output)
    return render_template("search.html")


if __name__=='__main__':
    app.run(port=8000,debug=True)
