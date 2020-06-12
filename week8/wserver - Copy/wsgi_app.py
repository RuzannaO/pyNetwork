from typing import Iterable
from configparser import ConfigParser

import urllib.parse
ROUTES={'/home': '<h1>Welcome to the first page!!</h1>','/second_page':'<h1>This is the second page.</h1>'}

class myapp1:
    def __init__(self):
        self.attr=155
    def __call__(self,env:dict, start_response:callable)-> Iterable:
        path=env["PATH_INFO"]
        if path in ROUTES:
            start_response('200 ok', [])
            return [ROUTES[path].encode()]
        start_response('404 NOT FOUND', [])
        return[b'Nothing found']


class myapp2:
    def __init__(self):
        self.attr=155155
    def __call__(self,env:dict, start_response:callable)-> Iterable:
        start_response('200 ok', [])
        return [b'This is my_app2']