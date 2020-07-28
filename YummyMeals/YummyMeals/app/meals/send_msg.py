from flask import Flask
from flask_mail import Mail, Message

app = Flask(__name__)
mail= Mail(app)

app.config['MAIL_SERVER']='smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = 'yummymealbook@gmail.com'
app.config['MAIL_PASSWORD'] = '789456123yummy'
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True


@app.route("/")
def send_mail():
   msg = Message('Hello', sender = 'yummymealbook@gmail.com', recipients = ['rordyan@gmail.com'])
   msg.body = """Hello dear user!   
            
We have good news for you.   A new recipe for your selected category has arrived!" 
              
Don't miss it!"""
   mail.send(msg)
   return "Sent"




#
if __name__ == '__main__':
   app.run(debug = True)