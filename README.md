---
title: PEC AnswerHub
emoji: 🌍
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.45.1
app_file: app.py
pinned: false
license: mit
short_description: 'PEC AI-powered policies assistant'
---


An AI-powered assistant for PEC registration queries based on official documents.

Setup:
Clone the repository: git clone https://github.com/asifarif/PEC-AnswerHub.git
Create a virtual environment: python3.10 -m venv venv
Activate the environment: source venv/bin/activate (Linux/Mac) or venv\Scripts\activate (Windows)
Install dependencies: pip install -r requirements.txt
Create .env with GROQ_API_KEY (get from https://x.ai/api).

Customizing the Frontend:
PEC Logo:
Replace the placeholder base64 string in app.py (PEC_LOGO_BASE64) with the actual PEC logo.
Convert the logo (PNG/JPG) to base64 using an online tool (e.g., https://www.base64-image.de/).
Update the src attribute in the <img> tag in app.py.

Styling:
Edit style.css to modify colors, fonts, or layout.
Ensure style.css is in the project root and loaded in app.py.

Run: streamlit run app.py

Deployment:
Deployed on Hugging Face Spaces via GitHub Actions.
Requires GitHub Secrets: HF_TOKEN (Hugging Face Write token), GROQ_API_KEY.

Usage
Ask questions like:
"What are the PEC registration requirements for contractors?"
"How can an engineering graduate register with PEC?"

Troubleshooting:
Check app.log for errors.
Ensure Google Drive links in policy_links.json are accessible.
Monitor Grok API usage at https://x.ai/api.