from typing import Iterable
from configparser import ConfigParser


ROUTES={'/home': '<h1>Welcome to the first page!!</h1>','/second_page':'<h1>This is the second page.</h1>'}

class myapp3:
    def __init__(self):
        self.title="my_app3"
    def __call__(self,env:dict, start_response:callable)-> Iterable:
        path=env["PATH_INFO"]
        if path in ROUTES:
            start_response('200 ok', [])
            return [ROUTES[path].encode()]
        start_response('404 NOT FOUND', [])
        return[b'Nothing found there']


class myapp4:
    def __init__(self):
        self.title = "my_app4"
    def __call__(self,env:dict, start_response:callable)-> Iterable:
        start_response('200 ok', [])
        return [b'This is my_app4']