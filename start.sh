#!/usr/bin/env bash

# Create the .streamlit directory
mkdir -p .streamlit

# Move the secrets file from the root to the .streamlit folder
# Render will place the secrets.toml file in the /app root directory
mv /app/secrets.toml /app/.streamlit/secrets.toml

# Run the Streamlit app
streamlit run app.py
