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
from google import genai
from google.genai import types
import re
from typing import Optional, Dict

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "your_secret_key"
limiter = Limiter(get_remote_address, app=app, default_limits=["5 per minute"])  # Limit to 5 requests per minute

# Contact form class
class ContactForm(FlaskForm):
    first_name = StringField("First Name", validators=[DataRequired()])
    last_name = StringField("Last Name", validators=[DataRequired()])
    email = StringField("Email", validators=[DataRequired(), Email()])
    message = TextAreaField("Message", validators=[DataRequired()])
    submit = SubmitField("Send Message")


def send_email(first_name, last_name, email, message):
    sender_email = "itimdcook@gmail.com"               # Your Gmail address
    sender_password = "jadm hlry qhqz przu"        # Your app-specific Gmail password
    receiver_email = "neerajshetkar@gmail.com"
    subject = f"New Contact Form Submission from {first_name} {last_name}"
    
    # Create HTML content with pretty CSS
    html = f"""\
    <html>
      <head>
        <style>
          body {{
            font-family: Arial, sans-serif;
            background-color: #f9f9f9;
            margin: 0;
            padding: 20px;
          }}
          .container {{
            background-color: #ffffff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
          }}
          h2 {{
            color: #333333;
          }}
          p {{
            color: #555555;
            font-size: 16px;
          }}
          .label {{
            font-weight: bold;
          }}
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
    
    # Create the email message
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = receiver_email
    
    # Attach the HTML part
    msg.attach(MIMEText(html, "html"))
    
    # Send the email using Gmail's SMTP server
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.ehlo()
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
    except Exception as e:
        # Log the error appropriately in production
        print(f"Error sending email: {e}")


@app.route("/contact/", methods=["POST"])
@limiter.limit("5 per minute")  # Apply rate limiting
def contact():
    form = ContactForm()
    
    if form.validate_on_submit():
        first_name = form.first_name.data
        last_name = form.last_name.data
        email = form.email.data
        message = form.message.data

        # Send the formatted HTML email
        send_email(first_name, last_name, email, message)
        
        flash("Your message has been sent successfully!", "success")
        return redirect(url_for("home"))
    
    return render_template("contact.html", form=form)


@app.route('/')
def home():
    return render_template('index.html', form=ContactForm())


def safe_json_extract(response: str) -> Optional[Dict]:
    """Robust JSON extraction with parsing"""
    try:
        # Try to find JSON between ``` markers
        json_match = re.search(r'```json(.*?)```', response, re.DOTALL) or re.search(r'```(.*?)```', response, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            # Fallback to finding first complete JSON object
            json_str = response[response.find('{'):response.rfind('}')+1]

        # Clean JSON string
        json_str = json_str.replace('\\"', '"')
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
        return json_str
    
    except (AttributeError, json.JSONDecodeError, KeyError) as e:
        print(f"JSON extraction error: {str(e)}")
        return None
    

def generate(description):

    prompt = '''
        Analyze this chemical compound description and return structural data in JSON format. Whatever be the strength of data
        show full representation of the compound in 3D space with all the atoms and bonds. Get all the data do not skip any element.
        Think over the resources and find correct data.
        Follow this EXACT structure:
        {
        "name": "IUPAC name",
        "properties": "Brief chemical description",
        "description": "Detailed description of the compound how its synthesized and what are its uses?",
        "formula": "Molecular formula",
        "atoms": {
            "C1": {
            "element": "C",
            "atomic_number": 6,
            "position": [x,y,z],
            "valence_electrons": 4,
            "hybridization": "sp3"
            },
            "O2": {
            "element": "O",
            "atomic_number": 8,
            "position": [x,y,z],
            "valence_electrons": 6,
            "hybridization": "sp2"
            },
            ...
        },
        "bonds": [
            {
            "atom1": "C1",
            "atom2": "C2",
            "bond_type": "single|double|triple",
            "plane": "horizontal|vertical",
            "angle": radians,
            "length": angstroms
            },
            ...
        ],
        "functional_groups": ["carboxylic acid", ...],
        "molecular_geometry": {
            "shape": "tetrahedral|trigonal-planar|etc",
            "bond_angles": [
            {
                "atoms": ["C1", "C2", "O1"],
                "degrees": 120.0
            },
            ...
            ]
        }
        }

        Important rules:
        1. Give unique IDs to atoms (e.g., C1, C2, O1, H1, H2)
        2. Positional co-ordinates be in range 0 to 0.75
        4. List all relevant bonds and bond angles
        5. Add a clear chemical description
        6. Include all the relevant functional groups
        8. Show all the elements and their positions
        9. Do not truncate any data
        10. Do not return anything other than JSON
    '''

    answer = ""

    client = genai.Client(
        api_key = "AIzaSyAb4TTvJNOcSeZe4BgwvUrBgUQeAoYvNXI",
    )

    model = "gemini-2.0-flash-lite"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt+description),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=1.5,
        top_p=0.95,
        top_k=40,
        max_output_tokens=16834,
        response_mime_type="application/json",
    )

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        answer += chunk.text + ""

    return answer


# Structured prompt template
def get_compound_data(description: str) -> dict:

    prompt = '''
        Analyze this chemical compound description and return structural data in JSON format. Whatever be the strength of data
        show full representation of the compound in 3D space with all the atoms and bonds. Get all the data do not skip any element.
        Follow this EXACT structure:
        {
        "name": "IUPAC name",
        "description": "Brief chemical description",
        "formula": "Molecular formula",
        "atoms": {
            "C1": {
            "element": "C",
            "atomic_number": 6,
            "position": [x,y,z],
            "valence_electrons": 4,
            "hybridization": "sp3"
            },
            "O2": {
            "element": "O",
            "atomic_number": 8,
            "position": [x,y,z],
            "valence_electrons": 6,
            "hybridization": "sp2"
            },
            ...
        },
        "bonds": [
            {
            "atom1": "C1",
            "atom2": "C2",
            "bond_type": "single|double|triple",
            "plane": "horizontal|vertical",
            "angle": radians,
            "length": angstroms
            },
            ...
        ],
        "functional_groups": ["carboxylic acid", ...],
        "molecular_geometry": {
            "shape": "tetrahedral|trigonal-planar|etc",
            "bond_angles": [
            {
                "atoms": ["C1", "C2", "O1"],
                "degrees": 120.0
            },
            ...
            ]
        },
        "description": "Detailed description of the compound how its synthesized and what are its uses?"
        }

        Important rules:
        1. Give unique IDs to atoms (e.g., C1, C2, O1, H1, H2)
        2. Specify exact atom positions in 3D space
        3. Include bond lengths in angstroms
        4. List all relevant bond angles
        5. Add a clear chemical description
        6. Include all the relevant functional groups
        8. Show all the elements and their positions
        9. Do not truncate any data
        10. Do not return anything other than JSON

        Compound description:
    '''
    
    try:
        response = generate(prompt + description)
        json_str = safe_json_extract(response)
        return json_str
    except json.JSONDecodeError:
        print("Failed to parse JSON response")
        return None
    except Exception as e:
        print(f"API Error: {str(e)}")
        return None


@app.route('/api/model', methods=['POST'])
def model():
    prompt = request.json
    response = get_compound_data(prompt["text"])
    print(response)
    return response

if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)