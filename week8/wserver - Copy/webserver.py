from wsgiref.simple_server import make_server
from configparser import ConfigParser
from wsgi_app import myapp1,myapp2
from wsgi_app_other import myapp3,myapp4


if __name__=='__main__':
    a=ConfigParser()
    a.read("config.ini")
    port=int(a["inet"]["port"])
    module=a["app"]["module"]
    app=a["app"]["app"]
    x = globals()[app]
    with make_server('',port,x()) as server:
        print(f'Serving on {port} , with module {module}  application {app}.....')
        server.serve_forever()