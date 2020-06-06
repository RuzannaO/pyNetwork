from urllib.request import urlopen

class getit:

    def html(self):
        return  str(urlopen('http://www.aca.am/en').read())
    def get_comment(self):
        self.comment=self.html()[48:1802]
        return self.comment
    def engl_text(self):
        assert isinstance(self.get_comment(), str), "please enter a str type argument"
        assert len([x for x in self.get_comment() if x not in "10 "]) == 0, "must contain only 0, 1 or 'whitespace'"
        result = ""
        for i in self.comment.split(' '):
            # print(chr(int(i,2)).encode())
            assert int(i, 2)<=255,"must be between 0-255 to be converted by ASCII"
            result += chr(int(i, 2))
        return result



n=getit()
print(n.get_comment())
print(n.engl_text())
