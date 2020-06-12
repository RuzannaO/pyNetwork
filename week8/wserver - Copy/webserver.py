# import importlib
# import sys
from wsgiref.simple_server import make_server
from configparser import ConfigParser
from wsgi_app import myapp1,myapp2


if __name__=='__main__':
    a=ConfigParser()
    a.read("config.ini")
    port=int(a["inet"]["port"])
    app=a["inet"]["app"]
    x = globals()[app]
    server=make_server('',port,x())
    print(f'Serving on {port} , with application {app}.....')
    server.serve_forever()