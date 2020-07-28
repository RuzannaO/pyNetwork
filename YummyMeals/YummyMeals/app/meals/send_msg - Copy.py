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
mail = Mail(app)
def send_mail():
   msg = Message('Hello', sender = 'yummymealbook@gmail.com', recipients = ['rordyan@gmail.com'])
   msg.body = "Hello dear Ruzanna!   We have good news for you.   A new recipe for your selected category has arrived!" \
              "Don't miss it!"
   mail.send(msg)
   return "Sent"

@app.route("/")
send_mail




#
if __name__ == '__main__':
   app.run(debug = True,port=5090)
   send_mail()
