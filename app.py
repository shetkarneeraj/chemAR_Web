from flask import Flask, render_template, request, flash, redirect, url_for
from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField, SubmitField
from wtforms.validators import DataRequired, Email
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import json

# **Initialize Flask App**
app = Flask(__name__)
app.secret_key = "your_secret_key"
limiter = Limiter(get_remote_address, app=app, default_limits=["5 per minute"])

# **Contact Form Class**
class ContactForm(FlaskForm):
    first_name = StringField("First Name", validators=[DataRequired()])
    last_name = StringField("Last Name", validators=[DataRequired()])
    email = StringField("Email", validators=[DataRequired(), Email()])
    message = TextAreaField("Message", validators=[DataRequired()])
    submit = SubmitField("Send Message")

# **Send Email Function**
def send_email(first_name: str, last_name: str, email: str, message: str):
    """Send contact form submission via email."""
    sender_email = "itimdcook@gmail.com"  # Replace with your email
    sender_password = "jadm hlry qhqz przu"  # Replace with your app-specific password
    receiver_email = "neerajshetkar@gmail.com"  # Replace with recipient email
    subject = f"New Contact Form Submission from {first_name} {last_name}"
    
    html = f"""\
    <html>
      <head>
        <style>
          body {{ font-family: Arial, sans-serif; background-color: #f9f9f9; margin: 0; padding: 20px; }}
          .container {{ background-color: #ffffff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
          h2 {{ color: #333333; }}
          p {{ color: #555555; font-size: 16px; }}
          .label {{ font-weight: bold; }}
        </style>
      </head>
      <body>
        <div class="container">
          <h2>New Contact Form Submission</h2>
          <p><span class="label">First Name:</span> {first_name}</p>
          <p><span class="label">Last Name:</span> {last_name}</p>
          <p><span class="label">Email:</span> {email}</p>
          <p><span class="label">Message:</span><br>{message}</p>
        </div>
      </body>
    </html>
    """
    
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg.attach(MIMEText(html, "html"))
    
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.ehlo()
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
    except Exception as e:
        print(f"Error sending email: {e}")

# **Contact Route**
@app.route("/contact/", methods=["POST"])
@limiter.limit("5 per minute")
def contact():
    """Handle contact form submissions."""
    form = ContactForm()
    if form.validate_on_submit():
        first_name = form.first_name.data
        last_name = form.last_name.data
        email = form.email.data
        message = form.message.data
        send_email(first_name, last_name, email, message)
        flash("Your message has been sent successfully!", "success")
        return redirect(url_for("home"))
    return render_template("contact.html", form=form)

# **Index Route**
@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html', form=ContactForm())

# **Model API Route**
@app.route('/api/model', methods=['POST'])
def model():
    """API endpoint to generate chemical compound data."""
    prompt = request.json
    if prompt.get("code") == "chemar2602":
        return json.dumps('''{
            "name": "N/A",
            "properties": "N/A",
            "description": "An important update has been rolled out. Please update your application for better experience.",
            "formula": "N/A",
            "atoms": {},
            "bonds": [],
            "functional_groups": [],
            "molecular_geometry": {
                "shape": "N/A",
                "bond_angles": []
            }''')

    else:
        return "Invalid code"

if __name__ == '__main__':
   app.run(host="0.0.0.0", port=8000)