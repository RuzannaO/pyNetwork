from html.parser import HTMLParser
from urllib.request import urlopen
from bs4 import BeautifulSoup as BS

class links_check():

    def __init__(self,url):
        self.url=url
        self.addr_list=[]
        self.title_list=[]
        fd=urlopen(self.url).read()
        html=fd.decode()
        self.html=html
        self.soup=BS(html,"html.parser")

    def def_links(self):
        for x in self.soup.find_all("a",href=True,title=True):
            if x["href"][0]=="/" :
                self.title_list.append((f'{x["title"]}'))
                self.addr_list.append(f'{(self.url[:self.url.rfind("/")])}{x["href"][5:]}')
        self.title_list=[x for i, x in enumerate(self.title_list) if i == self.title_list.index(x)]
        self.addr_list = [x for i, x in enumerate(self.addr_list) if i == self.addr_list.index(x)]
        return self.title_list,self.addr_list


if __name__ =="__main__":

    url='https://en.wikipedia.org/wiki/History_of_Python'

    a=links_check(url)
    print(a.def_links()[1])
    print("-------------------------------------------------------------------------------------------------------")
    print(a.def_links()[0])
    print("--------------------------------------------------------------------------------------------------------")

    for i in range (0,len(a.addr_list)):
        x=links_check(a.addr_list[i])
        list=x.def_links()[0]
        print(f'{a.title_list[i]}')
        list_addr=x.def_links()[1]
        print(f'----------------------------------------------------------------------------------------------------')
        for j in range(0,len(list_addr)):

            n=links_check(list_addr[j])
            print(f'#{j+1}   {list[j]}')
            # print(n.def_links()[0])
        print("------------------------------------------------------------------------------------------------------")

