from pathlib import Path
from urllib.parse import parse_qs
from typing import Callable, Iterable
from wsgiref.simple_server import make_server
import json

class HTTPError(Exception):

    def __init__(self, reason: str, code: int):
        self.code = code
        self.reason = reason
        super().__init__(reason)

# checking if searched name is available in the stored list, and modifies the result into html text
def name_in_list(data_list:list,stored_data:list)->str:
    result = []
    for i in stored_data['person']:
        if data_list[0] == i['First name'][0] and data_list[1] == i["Last name"][0]:
             result.append([f'{i["First name"][0]}, {i["Last name"][0]},  {i["Sex"][0]}, {i["age"][0]}'])
    if len(result)==0:
        a=""
        output='Could not find the name in our list'
    else:
        output = 'Your search result:'
        a=(str(result)).strip("[]'")
        for i in a:
           if i in "]'" :
               a=a.replace(str(i),"")
           if i == "[":
               a=a.replace(i,"<p>")
    return f'<h3><p>{output}</p></h3>  <h4><i>{a}</i></h4>'


def get_user_data(env: dict):
    with open(Path('html/user_data.html'), 'rb') as fd:
        return fd.read()

def post_user_data(env: dict):
    expected_keys = ('First name','Last name','Sex','age')
    payload = env['wsgi.input'].read(int(env['CONTENT_LENGTH']))
    data = parse_qs(payload.decode())
    if len(data) != len(expected_keys):
        raise HTTPError('Bad Request', 400)
    for key in expected_keys:
        if key not in data:
            raise HTTPError('Bad Request', 400)
    with open('person.json') as json_file:
        persondata = json.load(json_file)
        persondata['person'].append(parse_qs(payload.decode()))
    with open('person.json', 'w') as f:
        json.dump(persondata, f, indent=4)
    print(persondata['person'][0])
    return get_user_data(env)

def get_search(env: dict):
    with open(Path('html/search.html'), 'rb') as fd:
        data=parse_qs(env["QUERY_STRING"])
        if len(data)==1:
            with open('person.json', 'r') as f:
                stored_data = json.load(f)
                print(stored_data, "from file")
                data_list=(data['search'])[0].split()
                if len(data_list)!=2:
                    raise HTTPError('Bad request',400)
                else:
                    return str(name_in_list(data_list,stored_data)).encode()
        else:
            return  fd.read()


def not_found(env: dict):
    raise HTTPError('Not Found', 404)


ROUTING_TABLE = {
    '/user_data': {
        'GET': get_user_data,
        'POST': post_user_data
    },
    '/user_data/send': {
        'POST': post_user_data
    },
    '/search': {
        'GET': get_search
    }
}

def app(env: dict, start_response: Callable) -> Iterable:
        # for key, val in env.items():
        #    print(key, '=', val)
        route = env['PATH_INFO']
        method = env['REQUEST_METHOD']
        try:
            handler = ROUTING_TABLE.get(route, {}).get(method, not_found)
            response = handler(env)
            start_response('200 OK', [('Content-type', 'text/html')])
            return [response]
        except HTTPError as herr:
            start_response(f'{herr.code} {herr.reason}', [('Content-type', 'text/html')])
            return [f'<h2>{herr.code} {herr.reason}</h2>'.encode()]

if __name__ == '__main__':
    with open('person.json', 'w') as f:  # writing JSON object
         json.dump({"person":[]}, f)

    with make_server('', 8000, app) as httpd:
        print('Serving on port 8000...')
        httpd.serve_forever()
