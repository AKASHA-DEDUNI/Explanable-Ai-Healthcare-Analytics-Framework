# pages/Help.py

import streamlit as st

st.title("🛠️ Help & Instructions")

st.markdown("""
### How to Use:
1. Upload a valid chest X-ray image (.png/.jpg).
2. Input patient **age** and **gender**.
3. Paste the **radiology report**.
4. Click **Predict and Explain**.

### Troubleshooting:
- Make sure the image isn't corrupted.
- Ensure the text is not empty.
- Restart app if you face errors.

### Contact:
If you need help, email: `your.email@example.com`
""")

